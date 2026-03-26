import os
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import (
    EndTrialRequest,
    EvalPolicy,
    GetBenchmarkRequest,
    StartPlaygroundRequest,
    StatusRequest,
)
from connectrpc.errors import ConnectError

from agent import run_agent

BITGN_URL = os.getenv("BENCHMARK_HOST") or "https://api.bitgn.com"
BENCHMARK_ID = os.getenv("BENCHMARK_ID") or "bitgn/pac1-dev"
MODEL_ID = os.getenv("MODEL_ID") or "claude-sonnet-4-6"
PARALLEL_TASKS = int(os.getenv("PARALLEL_TASKS") or "1")

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_CLR = "\x1B[0m"
CLI_BLUE = "\x1B[34m"

_print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def run_task(client: HarnessServiceClientSync, task) -> tuple[str, float] | None:
    safe_print(f"{'=' * 30} Starting task: {task.task_id} {'=' * 30}")

    trial = client.start_playground(
        StartPlaygroundRequest(
            benchmark_id=BENCHMARK_ID,
            task_id=task.task_id,
        )
    )
    safe_print(f"{CLI_BLUE}{trial.instruction}{CLI_CLR}\n{'-' * 80}")

    try:
        run_agent(MODEL_ID, trial.harness_url, trial.instruction)
    except Exception as exc:
        safe_print(exc)

    result = client.end_trial(EndTrialRequest(trial_id=trial.trial_id))
    if result.score >= 0:
        style = CLI_GREEN if result.score == 1 else CLI_RED
        explain = textwrap.indent("\n".join(result.score_detail), "  ")
        safe_print(f"\n{style}Score: {result.score:0.2f}\n{explain}\n{CLI_CLR}")
        return task.task_id, result.score
    return None


def main() -> None:
    task_filter = os.sys.argv[1:]
    scores = []

    try:
        client = HarnessServiceClientSync(BITGN_URL)
        print("Connecting to BitGN", client.status(StatusRequest()))

        res = client.get_benchmark(GetBenchmarkRequest(benchmark_id=BENCHMARK_ID))
        print(
            f"{EvalPolicy.Name(res.policy)} benchmark: {res.benchmark_id} "
            f"with {len(res.tasks)} tasks.\n{CLI_GREEN}{res.description}{CLI_CLR}"
        )
        if PARALLEL_TASKS > 1:
            print(f"Running with PARALLEL_TASKS={PARALLEL_TASKS}")

        tasks_to_run = [
            task for task in res.tasks
            if not task_filter or task.task_id in task_filter
        ]

        if PARALLEL_TASKS > 1:
            with ThreadPoolExecutor(max_workers=PARALLEL_TASKS) as executor:
                futures = {executor.submit(run_task, client, task): task for task in tasks_to_run}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            scores.append(result)
                    except Exception as exc:
                        safe_print(f"Task error: {exc}")
        else:
            for task in tasks_to_run:
                result = run_task(client, task)
                if result:
                    scores.append(result)

    except ConnectError as exc:
        print(f"{exc.code}: {exc.message}")
    except KeyboardInterrupt:
        print(f"{CLI_RED}Interrupted{CLI_CLR}")

    if scores:
        for task_id, score in scores:
            style = CLI_GREEN if score == 1 else CLI_RED
            print(f"{task_id}: {style}{score:0.2f}{CLI_CLR}")
        total = sum(score for _, score in scores) / len(scores) * 100.0
        print(f"FINAL: {total:0.2f}%")


if __name__ == "__main__":
    main()
