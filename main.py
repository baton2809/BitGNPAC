"""
BitGN PAC1 — главный цикл.

Архитектура Main → Analyzer → Versioner (как у победителей ERC3):
  - После каждого провала: Analyzer извлекает урок
  - Каждые 3 урока: Versioner переписывает их в компактные правила
  - Правила передаются в extra_hint для следующих задач

Дополнительно:
  - Preflight wiki fetch: AGENTS.md читается один раз и инжектируется в system prompt
  - Cerebras API: CEREBRAS_API_KEY → llama-3.3-70b @ 3k tok/sec (бесплатно)
"""

import json
import os
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.vm.pcm_connect import PcmRuntimeClientSync
from bitgn.vm.pcm_pb2 import ReadRequest
from bitgn.harness_pb2 import (
    EndTrialRequest,
    EvalPolicy,
    GetBenchmarkRequest,
    StartPlaygroundRequest,
    StartRunRequest,
    StartTrialRequest,
    SubmitRunRequest,
    StatusRequest,
)
from connectrpc.errors import ConnectError

from agent import run_agent, log_header, CLI_GREEN, CLI_RED, CLI_CLR, CLI_YELLOW, CLI_CYAN, CLI_BLUE

# ─── Config ────────────────────────────────────────────────────────────────────

BITGN_URL      = os.getenv("BENCHMARK_HOST")     or "https://api.bitgn.com"
BENCHMARK_ID   = os.getenv("BENCHMARK_ID")       or "bitgn/pac1-dev"
BITGN_API_KEY  = os.getenv("BITGN_API_KEY")      or ""
MODEL_ID       = os.getenv("MODEL_ID")           or "gpt-oss:20b"
PARALLEL_TASKS = int(os.getenv("PARALLEL_TASKS") or "1")

# OpenRouter: задайте OPENROUTER_API_KEY для использования облачных моделей (напр. qwen/qwen3.6-plus-preview)
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Cerebras: задайте CEREBRAS_API_KEY для использования llama-3.3-70b в analyzer/versioner
CEREBRAS_API_KEY  = os.getenv("CEREBRAS_API_KEY")
CEREBRAS_MODEL    = os.getenv("CEREBRAS_MODEL")  or "llama-3.3-70b"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# Модель для анализатора/версионера (если не Cerebras — та же что и агент)
ANALYZER_MODEL = os.getenv("ANALYZER_MODEL") or MODEL_ID

_print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# ─── Analyzer/Versioner clients ────────────────────────────────────────────────

def _make_analyzer_client() -> OpenAI:
    if CEREBRAS_API_KEY:
        return OpenAI(api_key=CEREBRAS_API_KEY, base_url=CEREBRAS_BASE_URL)
    return OpenAI()


def _analyzer_model() -> str:
    return CEREBRAS_MODEL if CEREBRAS_API_KEY else ANALYZER_MODEL


# ─── Preflight wiki fetch ──────────────────────────────────────────────────────

def fetch_wiki(harness_url: str) -> str:
    """
    Читает ВСЕ AGENTS.md из workspace — root и вложенные.
    Формирует многоуровневый контекст для system prompt.

    Иерархия: /AGENTS.md (глобальные правила) + /subdir/AGENTS.md (локальные уточнения).
    Nested AGENTS.md имеют более высокий приоритет для файлов своей директории.
    """
    from bitgn.vm.pcm_pb2 import FindRequest
    vm = PcmRuntimeClientSync(harness_url)
    sections: list[str] = []

    try:
        # Найти все AGENTS.md файлы в воркспейсе
        r = vm.find(FindRequest(root="/", name="AGENTS.md", type=1, limit=20))
        paths = sorted(r.items)  # сортировка: root сначала (/, потом /subdir/)
    except ConnectError as exc:
        safe_print(f"{CLI_YELLOW}[wiki] find AGENTS.md failed: {exc.message}{CLI_CLR}")
        paths = ["/AGENTS.md"]

    if not paths:
        paths = ["/AGENTS.md"]

    for path in paths:
        try:
            content = vm.read(ReadRequest(path=path)).content.strip()
            if content:
                depth = path.count("/") - 1  # 0 = root, 1 = subdir, etc.
                label = "Global rules" if depth == 0 else f"Local rules for {path.rsplit('/', 1)[0]}/"
                sections.append(f"### {label} ({path})\n{content}")
                safe_print(f"{CLI_CYAN}[wiki] {path} ({len(content)} chars){CLI_CLR}")
        except ConnectError as exc:
            safe_print(f"{CLI_YELLOW}[wiki] {path} not found: {exc.message}{CLI_CLR}")

    return "\n\n".join(sections)


# ─── Analyzer: extract lesson from failure ─────────────────────────────────────

def _json_short(d: dict) -> str:
    try:
        return json.dumps(d)[:80]
    except Exception:
        return str(d)[:80]


def extract_lesson(
    analyzer: OpenAI,
    task_instruction: str,
    action_log: list[dict],
    score: float,
    score_detail: list[str],
) -> str | None:
    """
    Analyzer: после провала задачи извлекает один конкретный урок.
    Возвращает строку ≤2 предложений или None.
    """
    if score >= 1.0:
        return None

    steps_summary = "\n".join(
        f"  {s['tool']}({_json_short(s['args'])}) → {s['result'][:120]}"
        for s in action_log[-12:]
    ) or "  (no actions logged)"

    detail_text = "\n".join(score_detail) if score_detail else "(no detail)"

    prompt = (
        "A PKM file-management agent scored 0 on a task. "
        "Extract ONE specific actionable lesson about what went wrong — "
        "focus on file format, exact field names, file paths, or logic error. "
        "Be concrete, ≤2 sentences. No intro, just the lesson.\n\n"
        f"Task: {task_instruction[:400]}\n\n"
        f"Last agent actions:\n{steps_summary}\n\n"
        f"Score detail from harness:\n{detail_text}\n\n"
        "Lesson:"
    )

    try:
        resp = analyzer.chat.completions.create(
            model=_analyzer_model(),
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=150,
        )
        lesson = (resp.choices[0].message.content or "").strip()
        if lesson:
            safe_print(f"{CLI_YELLOW}[analyzer] {lesson}{CLI_CLR}")
        return lesson or None
    except Exception as exc:
        safe_print(f"{CLI_YELLOW}[analyzer] failed: {exc}{CLI_CLR}")
        return None


# ─── Versioner: rewrite lessons into clean rules ───────────────────────────────

def run_versioner(
    analyzer: OpenAI,
    raw_lessons: list[str],
    current_hint: str,
) -> str:
    """
    Versioner: раз в N уроков переписывает накопленные знания в компактные правила.
    Это то что у победителей эволюционировало 80 поколений.
    """
    lessons_text = "\n".join(f"- {l}" for l in raw_lessons)

    prompt = (
        "You are a Versioner for a PKM file-management agent. "
        "Below are raw lessons from failed tasks and the current hint. "
        "Rewrite them into a compact, numbered list of rules (≤10 rules, ≤1 sentence each). "
        "Remove duplicates. Keep only actionable rules about file format, paths, and logic.\n\n"
        f"Current hint:\n{current_hint or '(none)'}\n\n"
        f"New raw lessons:\n{lessons_text}\n\n"
        "Output: numbered rules only, no intro text."
    )

    try:
        resp = analyzer.chat.completions.create(
            model=_analyzer_model(),
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=300,
        )
        new_hint = (resp.choices[0].message.content or "").strip()
        if new_hint:
            safe_print(f"{CLI_CYAN}[versioner] evolved hint ({len(raw_lessons)} lessons → {len(new_hint)} chars):{CLI_CLR}")
            safe_print(f"{CLI_CYAN}{new_hint}{CLI_CLR}")
        return new_hint or current_hint
    except Exception as exc:
        safe_print(f"{CLI_YELLOW}[versioner] failed: {exc} — keeping current hint{CLI_CLR}")
        return current_hint


# ─── Task runner ───────────────────────────────────────────────────────────────

def run_task(
    client: HarnessServiceClientSync,
    task,
    analyzer: OpenAI,
    wiki_cache: dict,
    lessons: list[str],
    current_hint: str,
    trial=None,  # pre-started trial (from start_trial); if None — use start_playground
) -> tuple | None:
    if trial is None:
        trial = client.start_playground(
            StartPlaygroundRequest(benchmark_id=BENCHMARK_ID, task_id=task.task_id)
        )
    log_header(task.task_id, trial.instruction)

    # Preflight: fetch wiki (cached per harness_url)
    url = trial.harness_url
    if url not in wiki_cache:
        wiki_cache[url] = fetch_wiki(url)
    wiki_content = wiki_cache[url]

    action_log: list[dict] = []
    stats: dict = {}
    try:
        action_log, stats = run_agent(
            MODEL_ID,
            url,
            trial.instruction,
            wiki_content=wiki_content,
            extra_hint=current_hint,
        )
    except Exception as exc:
        safe_print(f"{CLI_RED}run_agent exception: {exc}{CLI_CLR}")

    result       = client.end_trial(EndTrialRequest(trial_id=trial.trial_id))
    score_detail = list(result.score_detail) if result.score_detail else []

    if result.score >= 0:
        style    = CLI_GREEN if result.score == 1.0 else CLI_RED
        symbol   = "✓" if result.score == 1.0 else "✗"
        elapsed  = stats.get("elapsed_s", 0.0)
        tok      = stats.get("total_tok", 0)
        calls    = stats.get("llm_calls", 0)
        safe_print(
            f"\n{style}  {symbol} Score: {result.score:0.2f}  [{task.task_id}]"
            f"  {elapsed:.1f}s  {tok} tok  {calls} LLM calls{CLI_CLR}"
        )
        if score_detail:
            for line in score_detail:
                safe_print(f"    {CLI_YELLOW}{line}{CLI_CLR}")
        return task.task_id, result.score, score_detail, trial.instruction, action_log, stats

    return None


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    task_filter = sys.argv[1:]
    scores: list[tuple[str, float]] = []
    all_stats: list[tuple[str, float, dict]] = []  # (task_id, score, stats)

    # Self-evolving state
    raw_lessons:   list[str] = []   # сырые уроки от Analyzer
    current_hint:  str       = ""   # эволюционирующий hint от Versioner
    wiki_cache:    dict      = {}   # кэш AGENTS.md per harness_url
    # Versioner отключён: gpt-oss:20b галлюцинирует несуществующие типы действий.
    # Включить когда будет более мощная модель (gpt-oss:120b / cerebras).
    VERSIONER_EVERY = 999

    # OpenAI-compatible base URL setup (priority: OpenRouter > explicit OPENAI_BASE_URL > Ollama)
    if OPENROUTER_API_KEY:
        os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL
        os.environ["OPENAI_API_KEY"]  = OPENROUTER_API_KEY
    elif not os.getenv("OPENAI_BASE_URL") and os.getenv("OLLAMA_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ["OLLAMA_BASE_URL"].rstrip("/") + "/v1"
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "ollama"

    analyzer = _make_analyzer_client()

    try:
        harness = HarnessServiceClientSync(BITGN_URL)
        print(f"BitGN: {harness.status(StatusRequest())}")

        res = harness.get_benchmark(GetBenchmarkRequest(benchmark_id=BENCHMARK_ID))
        print(
            f"{EvalPolicy.Name(res.policy)} benchmark: {res.benchmark_id} "
            f"({len(res.tasks)} tasks)\n{CLI_GREEN}{res.description}{CLI_CLR}"
        )
        model_info = f"model={MODEL_ID}"
        if OPENROUTER_API_KEY:
            model_info += " | via=openrouter"
        if CEREBRAS_API_KEY:
            model_info += f" | analyzer+versioner=cerebras/{CEREBRAS_MODEL}"
        print(f"{model_info} | parallel={PARALLEL_TASKS}\n")

        tasks_by_id = {t.task_id: t for t in res.tasks}
        tasks_to_run = [
            t for t in res.tasks
            if not task_filter or t.task_id in task_filter
        ]

        # ── Leaderboard run (api_key set) vs playground ───────────────────
        if BITGN_API_KEY:
            run_name = f"run model={MODEL_ID}"
            run = harness.start_run(StartRunRequest(
                name=run_name,
                benchmark_id=BENCHMARK_ID,
                api_key=BITGN_API_KEY,
            ))
            print(f"{CLI_CYAN}[leaderboard] run_id={run.run_id} ({len(run.trial_ids)} trials){CLI_CLR}")
            try:
                for trial_id in run.trial_ids:
                    trial = harness.start_trial(StartTrialRequest(trial_id=trial_id))
                    task  = tasks_by_id.get(trial.task_id)
                    if task_filter and trial.task_id not in task_filter:
                        continue
                    r = run_task(harness, task, analyzer, wiki_cache, raw_lessons, current_hint, trial=trial)
                    if r is None:
                        continue
                    task_id, score, score_detail, instruction, action_log, stats = r
                    scores.append((task_id, score))
                    all_stats.append((task_id, score, stats))
                    if score < 1.0:
                        lesson = extract_lesson(analyzer, instruction, action_log, score, score_detail)
                        if lesson:
                            raw_lessons.append(lesson)
                            print(f"{CLI_YELLOW}[{len(raw_lessons)} lessons]{CLI_CLR}")
                            if len(raw_lessons) % VERSIONER_EVERY == 0:
                                print(f"{CLI_CYAN}[versioner] running...{CLI_CLR}")
                                current_hint = run_versioner(analyzer, raw_lessons, current_hint)
            finally:
                harness.submit_run(SubmitRunRequest(run_id=run.run_id, force=True))
                print(f"{CLI_CYAN}[leaderboard] run submitted → https://bitgn.com/l/pac1-dev{CLI_CLR}")

        elif PARALLEL_TASKS > 1:
            # Параллельный режим: без self-evolving
            print(f"Parallel mode: {len(tasks_to_run)} tasks × {PARALLEL_TASKS} workers")
            with ThreadPoolExecutor(max_workers=PARALLEL_TASKS) as executor:
                futures = {
                    executor.submit(run_task, harness, task, analyzer, {}, [], ""): task
                    for task in tasks_to_run
                }
                for future in as_completed(futures):
                    try:
                        r = future.result()
                        if r:
                            scores.append((r[0], r[1]))
                            all_stats.append((r[0], r[1], r[5]))
                    except Exception as exc:
                        safe_print(f"Task error: {exc}")
        else:
            # Последовательный режим: Main → Analyzer → Versioner
            for task in tasks_to_run:
                r = run_task(harness, task, analyzer, wiki_cache, raw_lessons, current_hint)
                if r is None:
                    continue

                task_id, score, score_detail, instruction, action_log, stats = r
                scores.append((task_id, score))
                all_stats.append((task_id, score, stats))

                # ── Analyzer: извлечь урок из провала ───────────────────────
                if score < 1.0:
                    lesson = extract_lesson(
                        analyzer, instruction, action_log, score, score_detail
                    )
                    if lesson:
                        raw_lessons.append(lesson)
                        print(f"{CLI_YELLOW}[{len(raw_lessons)} lessons]{CLI_CLR}")

                        # ── Versioner: переписать уроки в правила ────────────
                        if len(raw_lessons) % VERSIONER_EVERY == 0:
                            print(f"{CLI_CYAN}[versioner] running...{CLI_CLR}")
                            current_hint = run_versioner(analyzer, raw_lessons, current_hint)

    except ConnectError as exc:
        print(f"{CLI_RED}{exc.code}: {exc.message}{CLI_CLR}")
    except KeyboardInterrupt:
        print(f"{CLI_RED}Interrupted{CLI_CLR}")

    # ── Итоговая таблица ─────────────────────────────────────────────────────
    if all_stats:
        print(f"\n{'=' * 72}")
        print(f"  {'Task':<6}  {'Score':>5}  {'Time':>8}  {'Tokens':>7}  {'Calls':>5}")
        print(f"  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*5}")
        total_time = 0.0
        total_tok  = 0
        total_call = 0
        for task_id, score, st in all_stats:
            style  = CLI_GREEN if score == 1.0 else CLI_RED
            symbol = "✓" if score == 1.0 else "✗"
            t = st.get("elapsed_s", 0.0)
            k = st.get("total_tok", 0)
            c = st.get("llm_calls", 0)
            total_time += t
            total_tok  += k
            total_call += c
            print(
                f"  {style}{symbol} {task_id:<5}  {score:>5.2f}  {t:>7.1f}s"
                f"  {k:>7}  {c:>5}{CLI_CLR}"
            )
        print(f"  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*7}  {'─'*5}")
        passed = sum(1 for _, s, _ in all_stats if s == 1.0)
        avg    = sum(s for _, s, _ in all_stats) / len(all_stats) * 100.0
        print(
            f"  {'TOTAL':<6}  {avg/100:>5.2f}  {total_time:>7.1f}s"
            f"  {total_tok:>7}  {total_call:>5}"
        )
        print(f"{'=' * 72}")
        print(f"\n  FINAL: {avg:.2f}%  ({passed}/{len(all_stats)} tasks)")
        print(f"  Avg per task: {total_time/len(all_stats):.1f}s, {total_tok//len(all_stats)} tok")

    if current_hint:
        print(f"\n{CLI_CYAN}Final evolved hint:{CLI_CLR}\n{current_hint}")


if __name__ == "__main__":
    main()
