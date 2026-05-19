"""
Human-in-the-loop (HITL) approval for expensive SQL tool calls.

Triggers when LIMIT >= QUERY_APPROVAL_MIN_LIMIT (default 1000) on query_database,
summarize_sql_stats, or create_chart. Approved SQL is remembered for the session
(normalized hash) so repeat calls do not re-prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field

_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)
_SQL_TOOLS = frozenset({"query_database", "summarize_sql_stats", "create_chart"})


def approval_enabled_from_env() -> bool:
    return os.getenv("ENABLE_QUERY_APPROVAL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def approval_min_limit() -> int:
    try:
        return max(1, int(os.getenv("QUERY_APPROVAL_MIN_LIMIT", "1000")))
    except ValueError:
        return 1000


def sql_approval_key(sql: str) -> str:
    norm = " ".join(str(sql).strip().split())
    return hashlib.sha256(norm.encode()).hexdigest()[:24]


def extract_limit_from_sql(sql: str | None) -> int | None:
    if not sql:
        return None
    matches = list(_LIMIT_RE.finditer(str(sql)))
    if not matches:
        return None
    return int(matches[-1].group(1))


@dataclass(frozen=True)
class ApprovalRequest:
    """One tool call waiting for human approval."""

    tool_name: str
    function_args: dict
    tool_call_id: str
    reason: str
    limit_value: int | None
    sql_preview: str


@dataclass
class ApprovalCheckpoint:
    """State required to resume a paused turn after approval UI."""

    user_input: str
    effective_role: str
    tool_round: int
    sql_failures_this_turn: int
    analyst_intent: dict
    planner_text: str | None
    tool_events: list[str] = field(default_factory=list)
    chart_paths: list[str] = field(default_factory=list)
    last_successful_chart_result: dict | None = None
    pending_calls: list[dict] = field(default_factory=list)
    assistant_reasoning: str | None = None
    planner_disabled: bool = False
    planner_temperature: float = 0.2
    planner_max_tokens: int = 600
    executor_temperature: float = 0.2
    executor_max_tokens: int | None = None
    query_approval_enabled: bool = True


def check_approval_required(
    tool_name: str,
    arguments: dict,
    *,
    enabled: bool,
    min_limit: int | None = None,
    approved_keys: set[str] | None = None,
) -> ApprovalRequest | None:
    """
    Return ApprovalRequest if this tool call must pause for human approval.
    """
    if not enabled or tool_name not in _SQL_TOOLS:
        return None
    threshold = min_limit if min_limit is not None else approval_min_limit()
    sql = arguments.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        return None
    limit_val = extract_limit_from_sql(sql)
    if limit_val is None or limit_val < threshold:
        return None
    key = sql_approval_key(sql)
    if approved_keys and key in approved_keys:
        return None
    preview = sql.strip()
    if len(preview) > 1200:
        preview = preview[:1200] + "\n… [truncated]"
    return ApprovalRequest(
        tool_name=tool_name,
        function_args=dict(arguments),
        tool_call_id="",
        reason=(
            f"SQL LIMIT {limit_val} is at or above the approval threshold ({threshold}). "
            "Confirm before running against the database."
        ),
        limit_value=limit_val,
        sql_preview=preview,
    )


def first_call_needing_approval(
    tool_calls: list,
    *,
    enabled: bool,
    approved_keys: set[str] | None = None,
) -> tuple[ApprovalRequest | None, list[dict]]:
    """
    Inspect a round of tool calls; return the first approval request and serialized pending calls.
    """
    pending: list[dict] = []
    first_req: ApprovalRequest | None = None
    for tc in tool_calls:
        fn = tc.function.name
        args = json.loads(tc.function.arguments)
        pending.append(
            {
                "id": tc.id,
                "name": fn,
                "arguments": args,
            }
        )
        if first_req is None:
            req = check_approval_required(
                fn, args, enabled=enabled, approved_keys=approved_keys
            )
            if req is not None:
                first_req = ApprovalRequest(
                    tool_name=req.tool_name,
                    function_args=req.function_args,
                    tool_call_id=tc.id,
                    reason=req.reason,
                    limit_value=req.limit_value,
                    sql_preview=req.sql_preview,
                )
    return first_req, pending
