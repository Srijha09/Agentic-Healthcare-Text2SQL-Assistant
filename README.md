# Labs@ Assessment

A terminal-based chat application that uses OpenAI's GPT-4o with function calling against a healthcare DuckDB database (synthetic data).

## Features

- Interactive chat (`chat.py`) or **Streamlit** UI (`streamlit_app.py`); full message history with a configurable system prompt (rolling context includes the **latest user request** to ground suggested follow-ups)
- GPT-4o + tools: `list_tables`, `query_database`, `describe_table`, `table_info`, `profile_table`, `create_chart`
- Optional **planner** step (one completion without tools), then **executor** loop with multi-round tool calls; `DISABLE_PLANNER=1` skips the planner call
- Markdown export: type `export`, or `exit` / Ctrl+C — writes `outputs/reports/session_<timestamp>.md` with a **reproducibility** block (model, optional git head, DB file size/mtime), planner text, SQL, redacted tool I/O, final answer
- SQL: `SELECT ... FROM ...` must include **`LIMIT n`** (enforced in [`tools/sql_guard.py`](tools/sql_guard.py)); long-running queries can hit a **wall-clock timeout** (DuckDB interrupt)
- SQL guardrail: after two failed `query_database` results in one user turn, further SQL for that turn returns `retry_limit_reached`

## How it works

| Piece | Where |
|-------|--------|
| Planner then executor | [`agent_orchestrator.py`](agent_orchestrator.py) (used by [`chat.py`](chat.py) and [`streamlit_app.py`](streamlit_app.py)): planner (no tools), then tools + answer |
| Tool definitions + dispatch | [`tools/tool_registry.py`](tools/tool_registry.py) — `dispatch_tool` routes all SQL through [`tools/db_query.py`](tools/db_query.py) |
| Cross-turn context | [`tools/session_state.py`](tools/session_state.py) — latest user request snippet, last SQL, tables, chart paths (system prompt) |
| Session export | [`session_log.py`](session_log.py) — export redacts long numeric tokens in SQL and `PATIENT_NUMBER` cells in JSON |
| SQL policy + timed execution | [`tools/sql_guard.py`](tools/sql_guard.py), used by `query_database` and `create_chart` |
| Charts | [`tools/chart_tool.py`](tools/chart_tool.py) → `outputs/visualization/` |

**Example SQL (diabetes / metformin / LAB gap / provider cost):** [`examples/workflow_flagship_diabetes_analytics.md`](examples/workflow_flagship_diabetes_analytics.md). **Time-to-event (dx signal to metformin):** [`examples/time_to_event_sql.md`](examples/time_to_event_sql.md). **Cost + chart patterns:** [`examples/bundle_c_cost_and_chart.md`](examples/bundle_c_cost_and_chart.md). **Cohort/trend pointer:** [`examples/bundle_a_cohort_trend.md`](examples/bundle_a_cohort_trend.md).

## Assessment basics (Phase 1)

| Requirement | How it is met |
|-------------|----------------|
| **LLM tool/function calling** | OpenAI Chat Completions with `tools` in [`tools/tool_registry.py`](tools/tool_registry.py), wired from [`chat.py`](chat.py). |
| **Healthcare database** | Read-only queries via `DuckDBQuery` in [`tools/db_query.py`](tools/db_query.py) (`healthcare.duckdb`). |
| **Agent orchestration** | Planner completion without tools, then executor with multi-round tools before the final message. |
| **Clean code** | `dispatch_tool` in [`chat.py`](chat.py); export in [`session_log.py`](session_log.py); structured JSON tool results. |

`tools/db_query.py` is used as provided and not modified.

## Database tools

- **list_tables** — List tables
- **query_database** — Run SQL
- **describe_table** / **table_info** — Schema and samples
- **profile_table** — DuckDB `SUMMARIZE` on a capped sample
- **create_chart** — Two-column SQL (same `sql_guard` as `query_database`) + required **title** (cohort, metric, time/breakdown) → PNG under `outputs/visualization/`

## Available tables

**Patient / events:** demographics, geography, mortality, mx_events, rx_events  

**Code lookups:** icd10_codes, procedure_codes, ndc_products  

Join on `PATIENT_NUMBER`.

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)
- OpenAI API key ([platform.openai.com](https://platform.openai.com/api-keys))

## Setup

### Quick (macOS/Linux with `make` and `unzip`)

```bash
make all
```

### Windows (PowerShell)

```powershell
uv sync
Expand-Archive -Path "data\input.zip" -DestinationPath "data" -Force
uv run python data\generate_data.py
uv run python scripts\load_archives_to_duckdb.py
```

If `generate_data.py` fails on NumPy `int32` bounds, use `dtype=np.int64` for large `randint` (see `data/generate_data.py`).

### Manual

1. `make install` or `uv sync`
2. `make generate-data` (CSVs in `data/`)
3. `make setup-db` or `uv run python scripts/load_archives_to_duckdb.py` → `healthcare.duckdb`
4. Copy `env.template` to `.env` and set `OPENAI_API_KEY`

### Makefile

- `make install` — dependencies
- `make generate-data` — synthetic CSVs
- `make setup-db` — load DB
- `make clean-db` / `make test-db` / `make all` / `make help`

## Usage

```bash
uv run chat.py
```

**Web UI (Streamlit):** same agent stack; sidebar has audit export (Bundle A) and settings; charts show in-chat with a per-plot download button.

```bash
uv run streamlit run streamlit_app.py
```

**Commands:** normal chat; **`export`** writes session markdown to `outputs/reports/`; **`exit`** / **`quit`** ends (and exports if there is content); **Ctrl+C** same as exit.

### Example queries

- "What tables are available?"
- "How many distinct patients in demographics?"
- "Show monthly metformin fills for patients with an E11 admit diagnosis" (see workflow doc)

### Sample terminal output

```
Connected to healthcare database
Available tables: demographics, geography, icd10_codes, ...

OpenAI Chat Terminal (gpt-4o)
Orchestration: planner -> executor (tools) -> final answer
Commands: 'export' - save session ...  |  'exit' / 'quit' - quit
------------------------------------------------------------

You: How many patients are in the database?

[Orchestration] Tool round 1 — 1 call(s)
  -> query_database(sql=...)

Assistant: ...
```

## Design notes

- **Export:** Markdown for review/diffs; SQL failure cap enforced in code; no edits to `db_query.py`.
- **Registry:** Tool schemas and handlers live in `tool_registry.py`. Session context in the system prompt; suggested follow-ups via prompt (no extra API call).
- **Charts:** matplotlib non-interactive backend; chart tool expects two result columns.

## Possible next steps

- Pytest for `dispatch_tool` and export shape
- Streaming for long answers
- Stricter SQL validation before execution

## Demo / examples

- [`examples/workflow_flagship_diabetes_analytics.md`](examples/workflow_flagship_diabetes_analytics.md)
- [`examples/bundle_c_cost_and_chart.md`](examples/bundle_c_cost_and_chart.md)
- [`examples/bundle_a_cohort_trend.md`](examples/bundle_a_cohort_trend.md)

Session audit markdown under `outputs/reports/` and chart PNGs under `outputs/visualization/` are gitignored; generate locally.
