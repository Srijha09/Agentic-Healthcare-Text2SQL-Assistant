"""
Structured logging and lightweight tracing for the agent stack.

Environment:
  LOG_LEVEL          DEBUG | INFO | WARNING | ERROR (default INFO)
  LOG_FORMAT         text | json (default text)
  LOG_FILE           Optional path; also defaults to outputs/logs/agent.log when LOG_TO_FILE=1
  LOG_TO_FILE        1 / true / yes to append to outputs/logs/agent.log

Each user turn gets a turn_id; a session trace_id is created once per process (or per
configure_logging call). Log records include trace_id and turn_id when a turn is active.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_TRACE_ID: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
_TURN_ID: contextvars.ContextVar[str] = contextvars.ContextVar("turn_id", default="")
_SPAN_STACK: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "span_stack", default=()
)

_CONFIGURED = False
_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_LOG_DIR = _PROJECT_ROOT / "outputs" / "logs"


class _ContextFilter(logging.Filter):
    """Inject trace_id, turn_id, and active span into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _TRACE_ID.get() or "-"  # type: ignore[attr-defined]
        record.turn_id = _TURN_ID.get() or "-"  # type: ignore[attr-defined]
        stack = _SPAN_STACK.get()
        record.span = stack[-1] if stack else "-"  # type: ignore[attr-defined]
        return True


class _JsonFormatter(logging.Formatter):
    """One JSON object per line for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "-"),
            "turn_id": getattr(record, "turn_id", "-"),
            "span": getattr(record, "span", "-"),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = "".join(
                traceback.format_exception(*record.exc_info)
            ).strip()
        # Merge structured fields from Logger.log(..., extra={...})
        skip = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "taskName",
            "trace_id",
            "turn_id",
            "span",
        }
        for key, val in record.__dict__.items():
            if key not in skip and not key.startswith("_"):
                payload[key] = val
        return json.dumps(payload, ensure_ascii=False, default=str)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        span = getattr(record, "span", "-")
        base = (
            f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} "
            f"{record.levelname:7} "
            f"[trace={getattr(record, 'trace_id', '-')} "
            f"turn={getattr(record, 'turn_id', '-')} "
            f"span={span}] "
            f"{record.name}: {record.getMessage()}"
        )
        if record.exc_info and record.exc_info[0] is not None:
            base += "\n" + "".join(traceback.format_exception(*record.exc_info))
        return base


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def configure_logging(
    *,
    level: str | None = None,
    log_format: str | None = None,
    log_file: str | Path | None = None,
) -> None:
    """
    Idempotent logging setup for chat, Streamlit, and tests.
    Call once at process entry (after load_dotenv).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    lvl_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    lvl = getattr(logging, lvl_name, logging.INFO)
    fmt_name = (log_format or os.getenv("LOG_FORMAT", "text")).strip().lower()

    root = logging.getLogger()
    root.setLevel(lvl)
    root.handlers.clear()

    ctx_filter = _ContextFilter()
    formatter: logging.Formatter
    if fmt_name == "json":
        formatter = _JsonFormatter()
    else:
        formatter = _TextFormatter()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(ctx_filter)
    root.addHandler(stream_handler)

    file_path: Path | None = None
    if log_file:
        file_path = Path(log_file)
    elif _env_truthy("LOG_TO_FILE"):
        file_path = _DEFAULT_LOG_DIR / "agent.log"

    if file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(ctx_filter)
        root.addHandler(file_handler)

    if not _TRACE_ID.get():
        _TRACE_ID.set(uuid.uuid4().hex[:12])

    _CONFIGURED = True
    get_logger("observability").info(
        "logging configured",
        extra={
            "event": "logging.configured",
            "log_level": lvl_name,
            "log_format": fmt_name,
            "log_file": str(file_path) if file_path else None,
            "trace_id": _TRACE_ID.get(),
        },
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def current_trace_id() -> str:
    return _TRACE_ID.get() or "-"


def current_turn_id() -> str:
    return _TURN_ID.get() or "-"


def start_turn(user_input: str | None = None) -> str:
    """Begin a new turn scope; returns turn_id."""
    turn_id = uuid.uuid4().hex[:10]
    _TURN_ID.set(turn_id)
    if not _TRACE_ID.get():
        _TRACE_ID.set(uuid.uuid4().hex[:12])
    snippet = (user_input or "").strip().replace("\n", " ")[:120]
    get_logger("agent.turn").info(
        "turn started",
        extra={
            "event": "turn.start",
            "turn_id": turn_id,
            "trace_id": _TRACE_ID.get(),
            "user_snippet": snippet,
            "user_chars": len(user_input or ""),
        },
    )
    return turn_id


def end_turn(
    *,
    status: str,
    error: str | None = None,
    tool_rounds: int | None = None,
    approval_pending: bool = False,
    peer_review_verdict: str | None = None,
) -> None:
    """Log turn completion; does not clear turn_id (safe for nested calls)."""
    extra: dict[str, Any] = {
        "event": "turn.end",
        "turn_id": _TURN_ID.get(),
        "trace_id": _TRACE_ID.get(),
        "status": status,
        "approval_pending": approval_pending,
    }
    if tool_rounds is not None:
        extra["tool_rounds"] = tool_rounds
    if peer_review_verdict is not None:
        extra["peer_review_verdict"] = peer_review_verdict
    if error:
        extra["error"] = error[:500]
    logger = get_logger("agent.turn")
    if status == "error":
        logger.error("turn finished with error", extra=extra)
    elif status in ("approval_pending", "interrupted"):
        logger.warning("turn finished: %s", status, extra=extra)
    else:
        logger.info("turn finished ok", extra=extra)


@contextmanager
def trace_span(name: str, **attrs: Any) -> Iterator[None]:
    """
    Nested span for orchestrator phases, tools, SQL, etc.
    Logs span.start / span.ok / span.fail with duration_ms.
    """
    logger = get_logger("agent.span")
    stack = _SPAN_STACK.get()
    _SPAN_STACK.set(stack + (name,))
    start = time.perf_counter()
    base_extra = {"event": "span.start", "span": name, **attrs}
    logger.debug("span start: %s", name, extra=base_extra)
    try:
        yield
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.exception(
            "span failed: %s (%s)",
            name,
            type(exc).__name__,
            extra={
                "event": "span.fail",
                "span": name,
                "duration_ms": round(elapsed_ms, 2),
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
                **attrs,
            },
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.debug(
            "span ok: %s (%.1f ms)",
            name,
            elapsed_ms,
            extra={
                "event": "span.ok",
                "span": name,
                "duration_ms": round(elapsed_ms, 2),
                **attrs,
            },
        )
    finally:
        _SPAN_STACK.set(stack)


def log_tool_result(tool_name: str, result_json: str, *, duration_ms: float | None = None) -> None:
    """Parse tool JSON and log success or structured failure (error_kind)."""
    logger = get_logger("agent.tool")
    extra: dict[str, Any] = {
        "event": "tool.result",
        "tool": tool_name,
    }
    if duration_ms is not None:
        extra["duration_ms"] = round(duration_ms, 2)
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        logger.warning(
            "tool returned non-JSON",
            extra={**extra, "parse_error": True, "preview": result_json[:200]},
        )
        return
    if not isinstance(data, dict):
        logger.info("tool ok (non-dict json)", extra=extra)
        return
    if "error" in data:
        extra["error_kind"] = data.get("error_kind", "unknown")
        extra["error"] = str(data.get("error", ""))[:300]
        if data.get("retry_limit_reached"):
            extra["retry_limit_reached"] = True
        logger.warning("tool failed: %s", tool_name, extra=extra)
        return
    if tool_name == "query_database":
        extra["total_rows"] = data.get("total_rows")
        extra["truncated"] = data.get("truncated")
    if tool_name == "create_chart":
        extra["chart_path"] = data.get("chart_path")
    logger.info("tool ok: %s", tool_name, extra=extra)


def log_sql_guard_outcome(
    *,
    ok: bool,
    message: str | None = None,
    duration_ms: float | None = None,
    row_count: int | None = None,
) -> None:
    logger = get_logger("agent.sql")
    extra: dict[str, Any] = {"event": "sql.execute"}
    if duration_ms is not None:
        extra["duration_ms"] = round(duration_ms, 2)
    if row_count is not None:
        extra["row_count"] = row_count
    if ok:
        logger.info("sql ok", extra=extra)
    else:
        extra["error"] = (message or "unknown")[:400]
        logger.warning("sql blocked or failed", extra=extra)
