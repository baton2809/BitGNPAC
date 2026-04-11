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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

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
BENCHMARK_ID   = os.getenv("BENCHMARK_ID")       or "bitgn/pac1-prod"
BITGN_API_KEY  = os.getenv("BITGN_API_KEY")      or ""
MODEL_ID       = os.getenv("MODEL_ID")           or "gpt-oss:20b"
PARALLEL_TASKS = int(os.getenv("PARALLEL_TASKS") or "1")

# OpenRouter: OPENROUTER_API_KEY, OPENROUTER_API_KEY_2 .. OPENROUTER_API_KEY_10 (любое кол-во)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY") or ""
_OPENROUTER_ALL_KEYS = [
    k for k in [
        OPENROUTER_API_KEY,
        *[os.getenv(f"OPENROUTER_API_KEY_{i}") or "" for i in range(2, 11)],
    ]
    if k
]

# Local fallback model (used when all OpenRouter keys are rate-limited)
# Set LOCAL_BASE_URL to your Ollama endpoint, LOCAL_MODEL_ID to the model name
_raw_ollama = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
LOCAL_MODEL_ID = os.getenv("LOCAL_MODEL_ID") or "gpt-oss:20b"
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL") or (
    (_raw_ollama + "/v1") if _raw_ollama else "http://localhost:11434/v1"
)

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


# ─── OpenRouter key pool ───────────────────────────────────────────────────────

class _KeyPool:
    """
    Round-robin pool of OpenAI clients for OpenRouter.
    Supports on_rate_limit(client): temporarily skips the rate-limited key
    for COOLDOWN seconds, returns the next available client immediately.
    When ALL OpenRouter keys are rate-limited, falls back to a local model client.
    """
    COOLDOWN = 60  # seconds to skip a rate-limited key

    def __init__(self, keys: list[str], local_base_url: str = "", local_model: str = ""):
        self._clients: list[OpenAI] = []
        self._cooldown_until: list[float] = []  # per-client cooldown timestamp
        for k in keys:
            self._clients.append(OpenAI(api_key=k, base_url=OPENROUTER_BASE_URL))
            self._cooldown_until.append(0.0)
        self._idx = 0
        self._lock = threading.Lock()
        if self._clients:
            safe_print(f"[key-pool] {len(self._clients)} OpenRouter key(s) loaded")

        # Local fallback: used when all OR keys are rate-limited or none provided
        self._local_client: OpenAI | None = None
        self._local_model: str = local_model
        if local_base_url:
            self._local_client = OpenAI(api_key="local", base_url=local_base_url)
            safe_print(f"[key-pool] local fallback: {local_base_url} ({local_model})")

    @property
    def local_model(self) -> str:
        return self._local_model

    def is_local(self, client: "OpenAI") -> bool:
        return self._local_client is not None and client is self._local_client

    def next(self) -> "OpenAI | None":
        """Return next available (non-rate-limited) client, or local fallback if all exhausted."""
        if not self._clients:
            return self._local_client  # pure-local mode
        with self._lock:
            now = time.time()
            n = len(self._clients)
            # Try each slot starting from current idx, skip rate-limited
            for _ in range(n):
                i = self._idx % n
                self._idx += 1
                if now >= self._cooldown_until[i]:
                    return self._clients[i]
            # All OpenRouter keys are rate-limited — use local fallback
            if self._local_client:
                safe_print("[key-pool] all OR keys exhausted — using local fallback")
                return self._local_client
            # No local fallback — return least-recently limited key (will retry)
            i = min(range(n), key=lambda x: self._cooldown_until[x])
            return self._clients[i]

    def on_rate_limit(self, client: "OpenAI") -> "OpenAI | None":
        """
        Mark client as rate-limited for COOLDOWN seconds.
        Returns the next available client (different from this one if possible).
        Falls back to local model if all OR keys become exhausted.
        """
        if not self._clients:
            return self._local_client
        # Local client can't be rate-limited — just return it again
        if client is self._local_client:
            return self._local_client
        with self._lock:
            now = time.time()
            n = len(self._clients)
            # Mark this client as cooled down
            for i, c in enumerate(self._clients):
                if c is client:
                    self._cooldown_until[i] = now + self.COOLDOWN
                    safe_print(f"[key-pool] key[{i}] rate-limited, cooldown {self.COOLDOWN}s")
                    break
            # Return next non-limited client
            for _ in range(n):
                j = self._idx % n
                self._idx += 1
                if now >= self._cooldown_until[j]:
                    return self._clients[j]
            # All OpenRouter keys exhausted — use local fallback
            if self._local_client:
                safe_print("[key-pool] all OR keys rate-limited — switching to local fallback")
                return self._local_client
            # No local fallback — return least-recently limited
            j = min(range(n), key=lambda x: self._cooldown_until[x])
            return self._clients[j]

    def __bool__(self):
        return bool(self._clients) or self._local_client is not None


_key_pool = _KeyPool(_OPENROUTER_ALL_KEYS, local_base_url=LOCAL_BASE_URL, local_model=LOCAL_MODEL_ID)

# ─── Shared wiki cache (thread-safe, double-checked locking) ──────────────────
_wiki_cache: dict[str, str] = {}
_wiki_lock = threading.Lock()


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
    lessons: list[str],
    current_hint: str,
    trial=None,  # pre-started trial (from start_trial); if None — use start_playground
) -> tuple | None:
    if trial is None:
        trial = client.start_playground(
            StartPlaygroundRequest(benchmark_id=BENCHMARK_ID, task_id=task.task_id)
        )
    log_header(task.task_id, trial.instruction)

    # Preflight: fetch wiki (shared cache, double-checked locking)
    url = trial.harness_url
    if url not in _wiki_cache:
        with _wiki_lock:
            if url not in _wiki_cache:  # double-check inside lock
                _wiki_cache[url] = fetch_wiki(url)
    wiki_content = _wiki_cache[url]

    action_log: list[dict] = []
    stats: dict = {}
    try:
        _initial_client = _key_pool.next() if _key_pool else None
        _initial_model  = (
            _key_pool.local_model
            if _key_pool and _initial_client and _key_pool.is_local(_initial_client)
            else MODEL_ID
        )
        action_log, stats = run_agent(
            _initial_model,
            url,
            trial.instruction,
            wiki_content=wiki_content,
            extra_hint=current_hint,
            openai_client=_initial_client,
            key_pool=_key_pool if _key_pool else None,
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
    raw_lessons:  list[str] = []   # сырые уроки от Analyzer
    current_hint: str       = ""   # эволюционирующий hint от Versioner
    # wiki_cache теперь глобальный _wiki_cache (thread-safe, shared across workers)
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
        harness = HarnessServiceClientSync(BITGN_URL, api_key=BITGN_API_KEY)
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
            print(f"{CLI_CYAN}[leaderboard] run_id={run.run_id} ({len(run.trial_ids)} trials) workers={PARALLEL_TASKS}{CLI_CLR}")

            # Filter trial_ids by task_filter if set (peek task_id without full start)
            # For competition: no filter → all trial_ids used
            trials_to_run = [
                tid for tid in run.trial_ids
                if not task_filter  # when task_filter set, we'll check after start_trial
            ]
            if task_filter:
                # Need to peek task_id — start_trial is the only way
                for tid in run.trial_ids:
                    tr = harness.start_trial(StartTrialRequest(trial_id=tid))
                    if tr.task_id in task_filter:
                        trials_to_run.append(tr)  # store full trial object
                trial_ids_to_run = trials_to_run  # mixed: str ids or trial objects
            else:
                trial_ids_to_run = trials_to_run  # list of str trial_ids

            TASK_TIMEOUT = 300  # 5 min max per task

            def _run_leaderboard_trial(tid_or_trial):
                if isinstance(tid_or_trial, str):
                    trial = harness.start_trial(StartTrialRequest(trial_id=tid_or_trial))
                else:
                    trial = tid_or_trial
                task = tasks_by_id.get(trial.task_id)
                return run_task(harness, task, analyzer, [], "", trial=trial)

            try:
                if PARALLEL_TASKS > 1:
                    with ThreadPoolExecutor(max_workers=PARALLEL_TASKS) as executor:
                        futures = {executor.submit(_run_leaderboard_trial, t): t for t in trial_ids_to_run}
                        for future in as_completed(futures):
                            try:
                                r = future.result(timeout=TASK_TIMEOUT)
                                if r:
                                    scores.append((r[0], r[1]))
                                    all_stats.append((r[0], r[1], r[5]))
                            except FutureTimeoutError:
                                safe_print(f"{CLI_RED}⚠ Trial timeout ({TASK_TIMEOUT}s) — skipping{CLI_CLR}")
                            except Exception as exc:
                                safe_print(f"Trial error: {exc}")
                else:
                    for tid_or_trial in trial_ids_to_run:
                        if isinstance(tid_or_trial, str):
                            trial = harness.start_trial(StartTrialRequest(trial_id=tid_or_trial))
                        else:
                            trial = tid_or_trial
                        task = tasks_by_id.get(trial.task_id)
                        r = run_task(harness, task, analyzer, raw_lessons, current_hint, trial=trial)
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
                print(f"{CLI_CYAN}[leaderboard] run submitted → https://bitgn.com/l/pac1-prod{CLI_CLR}")

        elif PARALLEL_TASKS > 1:
            # Параллельный режим: без self-evolving
            print(f"Parallel mode: {len(tasks_to_run)} tasks × {PARALLEL_TASKS} workers")
            with ThreadPoolExecutor(max_workers=PARALLEL_TASKS) as executor:
                futures = {
                    executor.submit(run_task, harness, task, analyzer, [], ""): task
                    for task in tasks_to_run
                }
                for future in as_completed(futures):
                    try:
                        r = future.result(timeout=300)
                        if r:
                            scores.append((r[0], r[1]))
                            all_stats.append((r[0], r[1], r[5]))
                    except FutureTimeoutError:
                        safe_print(f"{CLI_RED}⚠ Task timeout (300s) — skipping{CLI_CLR}")
                    except Exception as exc:
                        safe_print(f"Task error: {exc}")
        else:
            # Последовательный режим: Main → Analyzer → Versioner
            for task in tasks_to_run:
                r = run_task(harness, task, analyzer, raw_lessons, current_hint)
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
