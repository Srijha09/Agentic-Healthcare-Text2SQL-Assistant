"""
Context window management: token estimation and message pruning.

Uses character-based approximation (1 token ≈ 4 chars) — no extra dependency.
Prunes old turns when the message list approaches the model's context limit,
always preserving tool_call / tool response pairs so the OpenAI API never
receives an assistant message with tool_calls whose responses have been dropped.
"""
from __future__ import annotations

_CHARS_PER_TOKEN = 4
_GPT4O_CONTEXT_LIMIT = 128_000
_PRUNE_THRESHOLD = 0.80  # prune when estimated usage exceeds 80% of limit


def _msg_char_len(msg: dict) -> int:
    total = 0
    content = msg.get("content") or ""
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                total += len(str(part.get("text") or ""))
    else:
        total += len(str(content))
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        total += len(str(tool_calls))
    return total


def estimate_tokens(messages: list[dict]) -> int:
    """Approximate token count for a message list (1 token ≈ 4 chars)."""
    return sum(_msg_char_len(m) for m in messages) // _CHARS_PER_TOKEN


def needs_pruning(
    messages: list[dict],
    *,
    limit: int = _GPT4O_CONTEXT_LIMIT,
    threshold: float = _PRUNE_THRESHOLD,
) -> bool:
    return estimate_tokens(messages) > int(limit * threshold)


def prune_messages(
    messages: list[dict],
    *,
    session_digest: str | None = None,
) -> list[dict]:
    """
    Compress old turns to free context space while keeping the message list
    valid for the OpenAI API.

    Strategy:
    - The system message (index 0) is always kept verbatim.
    - The last 2 user turns (and all messages between them) are kept verbatim.
    - Everything before that is replaced with a single compact summary message
      injected as a system message so it informs the model without acting as a
      user turn.
    - Optional session_digest (cohort SQL, recent tables) is appended to the summary.

    If there are 2 or fewer user turns nothing is pruned — there is nothing safe
    to drop without losing the current conversation.
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    user_indices = [i for i, m in enumerate(non_system) if m.get("role") == "user"]

    if len(user_indices) <= 2:
        return messages

    # Keep the last 2 full user turns verbatim.
    keep_from = user_indices[-2]
    to_compress = non_system[:keep_from]
    to_keep = non_system[keep_from:]

    lines: list[str] = []
    for msg in to_compress:
        role = msg.get("role", "")
        if role == "user":
            text = str(msg.get("content") or "")[:300]
            lines.append(f"User: {text}")
        elif role == "assistant" and msg.get("content"):
            text = str(msg.get("content") or "")[:300]
            lines.append(f"Assistant: {text}")
        elif role == "tool":
            lines.append(f"Tool `{msg.get('name', '?')}`: result omitted.")

    summary_body = "\n".join(lines)
    if session_digest and session_digest.strip():
        summary_body += (
            "\n\n[Session digest preserved across pruning]\n"
            + session_digest.strip()[:4000]
        )

    summary: dict = {
        "role": "system",
        "content": (
            "[Earlier turns compressed to stay within context limit]\n"
            + summary_body
        ),
    }

    return system_msgs + [summary] + to_keep
