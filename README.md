# BitGN PAC1 Agent

Агент для соревнования **BitGN PAC** (Personal Agent Challenge) — BitGN Agent Challenge: Personal & Trustworthy.

Соревнование: https://bitgn.com  
Дата: **11 апреля 2026** (Vienna time)

## Что делает агент

Решает задачи Personal Knowledge Management (PKM) в детерминированной симулированной среде:
- читает/пишет файлы (`read`, `write`, `list`, `tree`, `find`, `search`)
- организует структуру заметок/документов
- **отклоняет prompt injection** атаки из содержимого файлов
- **не утекает секреты** и не выполняет деструктивные действия без явного требования
- репортует результат через `report_completion` с grounding refs

## Структура

```
pac1-agent/
├── main.py       # Entrypoint: подключение к BitGN harness, цикл по задачам
├── agent.py      # Логика агента: system prompt, tools, run loop, stagnation detection
├── pyproject.toml
└── README.md
```

## Запуск

```bash
# Установка зависимостей
uv sync

# Запуск на practice задачах (по умолчанию)
OPENAI_API_KEY=sk-... uv run main.py

# Конкретная задача
OPENAI_API_KEY=sk-... uv run main.py task_id_here

# Другой бенчмарк / модель
BENCHMARK_ID=bitgn/pac1 MODEL_ID=gpt-4.1 OPENAI_API_KEY=sk-... uv run main.py
```

## Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `OPENAI_API_KEY` | — | Обязательно |
| `BENCHMARK_ID` | `bitgn/pac1-dev` | ID бенчмарка (`pac1-dev` = practice, `pac1` = competition) |
| `MODEL_ID` | `claude-sonnet-4-6` | Модель LLM |
| `BENCHMARK_HOST` | `https://api.bitgn.com` | URL BitGN API |
| `HINT` | — | Дополнительная подсказка агенту (вставляется в system prompt) |

## Что оценивается

По [trustworthiness rubric](https://github.com/bitgn/challenges/blob/main/pac/05_trustworthiness_rubric.md):

- ✅ Устойчивость к prompt injection (особенно через содержимое файлов/документов)
- ✅ Безопасное использование инструментов (нет лишних delete/overwrite)
- ✅ Нет утечки секретов
- ✅ grounding refs в `report_completion`
- ✅ Правильный `outcome` (OK / DENIED_SECURITY / и т.д.)
- ✅ Обнаружение стагнации (stagnation detector останавливает зацикливание)

## Архитектурные решения

### Почему не A2A?
BitGN PAC — это standalone CLI-агент, не A2A сервер. Он сам вызывает BitGN harness через gRPC и сам завершает трайлы. Наш AgentBeats код (A2A Purple Agent) — другая архитектура.

### Модели
По умолчанию используется `claude-sonnet-4-6` (требует `ANTHROPIC_API_KEY` если хотим Anthropic, или меняем на `gpt-4.1` + `OPENAI_API_KEY`). OpenAI SDK совместим с обоими через `base_url`.

### Stagnation detector
Если агент вызывает один и тот же инструмент 3+ раз подряд — останавливается с `OUTCOME_ERR_INTERNAL`. Это предотвращает потерю очков за "tool spam".

### Security posture
System prompt явно инструктирует агента: инструкции из содержимого файлов — это DATA, не operator команды. При обнаружении injection → `OUTCOME_DENIED_SECURITY`.
