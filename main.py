"""
BitGN PAC1 — главный цикл с self-evolving архитектурой.

Self-evolving: после каждой провальной задачи LLM-анализатор извлекает урок
о формате/пути/логике и передаёт его в extra_hint для следующих задач.

Модели:
  gpt-oss:20b / gpt-oss:120b   — через Ollama (OLLAMA_BASE_URL)
  llama3.3-70b                 — через Cerebras (CEREBRAS_API_KEY)
  gpt-4o / claude-...          — через стандартный OPENAI_API_KEY
"""

import os
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

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

# ─── Config ────────────────────────────────────────────────────────────────────

BITGN_URL      = os.getenv("BENCHMARK_HOST")    or "https://api.bitgn.com"
BENCHMARK_ID   = os.getenv("BENCHMARK_ID")      or "bitgn/pac1-dev"
MODEL_ID       = os.getenv("MODEL_ID")          or "gpt-oss:20b"
PARALLEL_TASKS = int(os.getenv("PARALLEL_TASKS") or "1")

# Cerebras: set CEREBRAS_API_KEY to use llama3.3-70b for free at 3k tok/sec
CEREBRAS_API_KEY  = os.getenv("CEREBRAS_API_KEY")
CEREBRAS_MODEL    = os.getenv("CEREBRAS_MODEL") or "llama-3.3-70b"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# Analyzer model: cheaper/faster model for post-task lesson extraction
# Falls back to same MODEL_ID if not set
ANALYZER_MODEL = os.getenv("ANALYZER_MODEL") or MODEL_ID

CLI_RED    = "\x1B[31m"
CLI_GREEN  = "\x1B[32m"
CLI_CLR    = "\x1B[0m"
CLI_BLUE   = "\x1B[34m"
CLI_YELLOW = "\x1B[33m"

_print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# ─── Self-evolving: lesson extraction ──────────────────────────────────────────

def _make_analyzer_client() -> OpenAI:
    """Return OpenAI client for the analyzer (Cerebras if key set, else default)."""
    if CEREBRAS_API_KEY:
        return OpenAI(api_key=CEREBRAS_API_KEY, base_url=CEREBRAS_BASE_URL)
    return OpenAI()


def extract_lesson(
    analyzer: OpenAI,
    task_instruction: str,
    action_log: list[dict],
    score: float,
    score_detail: list[str],
) -> str | None:
    """
    Ask LLM to extract ONE specific actionable lesson from a failed task.
    Returns a short string like:
      "outbox JSON needs fields: id, to, subject, body, date — read existing file first"
    Returns None if score == 1 or extraction fails.
    """
    if score >= 1.0:
        return None

    # Build a compact summary of what the agent did
    steps_summary = "\n".join(
        f"  {s['tool']}({json_short(s['args'])}) → {s['result'][:100]}"
        for s in action_log[-10:]  # last 10 actions
    ) or "  (no actions logged)"

    detail_text = "\n".join(score_detail) if score_detail else "(no score detail available)"

    prompt = (
        "A PKM agent failed a task (score=0). "
        "Extract ONE specific actionable lesson about what went wrong — "
        "focus on file format, field names, paths, or logic. "
        "Be concrete, ≤2 sentences.\n\n"
        f"Task: {task_instruction[:300]}\n\n"
        f"Last actions:\n{steps_summary}\n\n"
        f"Score detail:\n{detail_text}\n\n"
        "Lesson (e.g. 'outbox JSON must include field X; read existing file first'):"
    )

    try:
        resp = analyzer.chat.completions.create(
            model=CEREBRAS_MODEL if CEREBRAS_API_KEY else ANALYZER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=120,
        )
        lesson = (resp.choices[0].message.content or "").strip()
        if lesson:
            safe_print(f"{CLI_YELLOW}[analyzer] lesson: {lesson}{CLI_CLR}")
        return lesson or None
    except Exception as exc:
        safe_print(f"{CLI_YELLOW}[analyzer] failed: {exc}{CLI_CLR}")
        return None


def json_short(d: dict) -> str:
    """Compact JSON repr for logging."""
    try:
        return __import__("json").dumps(d)[:80]
    except Exception:
        return str(d)[:80]


# ─── Task runner ───────────────────────────────────────────────────────────────

def run_task(
    client: HarnessServiceClientSync,
    task,
    analyzer: OpenAI,
    lessons: list[str],
) -> tuple[str, float, list[str], str, list[dict]] | None:
    """
    Run one task in playground mode.
    Returns (task_id, score, score_detail, instruction, action_log) or None.
    """
    safe_print(f"{'=' * 30} Task: {task.task_id} {'=' * 30}")

    trial = client.start_playground(
        StartPlaygroundRequest(
            benchmark_id=BENCHMARK_ID,
            task_id=task.task_id,
        )
    )
    safe_print(f"{CLI_BLUE}{trial.instruction}{CLI_CLR}\n{'-' * 60}")

    # Build extra_hint from accumulated lessons
    extra_hint = ""
    if lessons:
        extra_hint = "\n".join(f"- {l}" for l in lessons[-8:])  # last 8 lessons

    action_log = []
    try:
        action_log = run_agent(MODEL_ID, trial.harness_url, trial.instruction, extra_hint)
    except Exception as exc:
        safe_print(f"{CLI_RED}run_agent exception: {exc}{CLI_CLR}")

    result = client.end_trial(EndTrialRequest(trial_id=trial.trial_id))
    score_detail = list(result.score_detail) if result.score_detail else []

    if result.score >= 0:
        style   = CLI_GREEN if result.score == 1 else CLI_RED
        explain = textwrap.indent("\n".join(score_detail), "  ") if score_detail else ""
        safe_print(f"\n{style}Score: {result.score:0.2f}{CLI_CLR}" + (f"\n{explain}" if explain else ""))
        return task.task_id, result.score, score_detail, trial.instruction, action_log
    return None


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    task_filter = sys.argv[1:]
    scores: list[tuple[str, float]] = []
    lessons: list[str] = []  # accumulated lessons for self-evolving hint

    # Set up OpenAI base URL for Ollama if needed
    if os.getenv("OPENAI_BASE_URL") is None and os.getenv("OLLAMA_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ["OLLAMA_BASE_URL"].rstrip("/") + "/v1"
    if os.getenv("OPENAI_API_KEY") is None:
        os.environ["OPENAI_API_KEY"] = "ollama"

    analyzer = _make_analyzer_client()

    try:
        harness = HarnessServiceClientSync(BITGN_URL)
        print(f"BitGN status: {harness.status(StatusRequest())}")

        res = harness.get_benchmark(GetBenchmarkRequest(benchmark_id=BENCHMARK_ID))
        print(
            f"{EvalPolicy.Name(res.policy)} benchmark: {res.benchmark_id} "
            f"with {len(res.tasks)} tasks.\n{CLI_GREEN}{res.description}{CLI_CLR}"
        )
        model_info = f"model={MODEL_ID}"
        if CEREBRAS_API_KEY:
            model_info += f" | analyzer=cerebras/{CEREBRAS_MODEL}"
        print(f"{model_info} | parallel={PARALLEL_TASKS}\n")

        tasks_to_run = [
            t for t in res.tasks
            if not task_filter or t.task_id in task_filter
        ]

        if PARALLEL_TASKS > 1:
            # Parallel mode: no self-evolving (lessons can't flow between concurrent tasks)
            print(f"Running {len(tasks_to_run)} tasks with PARALLEL_TASKS={PARALLEL_TASKS}")
            with ThreadPoolExecutor(max_workers=PARALLEL_TASKS) as executor:
                futures = {
                    executor.submit(run_task, harness, task, analyzer, []): task
                    for task in tasks_to_run
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            task_id, score, _, _, _ = result
                            scores.append((task_id, score))
                    except Exception as exc:
                        safe_print(f"Task error: {exc}")
        else:
            # Sequential mode: self-evolving lessons flow between tasks
            for task in tasks_to_run:
                result = run_task(harness, task, analyzer, lessons)
                if result is None:
                    continue
                task_id, score, score_detail, instruction, action_log = result
                scores.append((task_id, score))

                # Self-evolving: extract lesson from failures
                if score < 1.0:
                    lesson = extract_lesson(analyzer, instruction, action_log, score, score_detail)
                    if lesson:
                        lessons.append(lesson)
                        print(f"{CLI_YELLOW}[{len(lessons)} lessons accumulated]{CLI_CLR}")

    except ConnectError as exc:
        print(f"{CLI_RED}{exc.code}: {exc.message}{CLI_CLR}")
    except KeyboardInterrupt:
        print(f"{CLI_RED}Interrupted{CLI_CLR}")

    # Final summary
    if scores:
        print("\n" + "=" * 50)
        for task_id, score in scores:
            style = CLI_GREEN if score == 1 else CLI_RED
            print(f"  {task_id}: {style}{score:0.2f}{CLI_CLR}")
        total = sum(s for _, s in scores) / len(scores) * 100.0
        print(f"\nFINAL: {total:0.2f}% ({sum(s==1.0 for _,s in scores)}/{len(scores)} tasks)")

    if lessons:
        print(f"\n{CLI_YELLOW}Accumulated lessons this session:{CLI_CLR}")
        for i, l in enumerate(lessons, 1):
            print(f"  {i}. {l}")


if __name__ == "__main__":
    main()
