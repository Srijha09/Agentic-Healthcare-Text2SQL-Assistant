"""
Read-only SQL → simple PNG charts (matplotlib, non-interactive backend).

All chart SQL runs through tools.sql_guard.query_with_columns_timed — same LIMIT rules and
timeout as query_database.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from tools.db_query import DuckDBQuery
from tools.sql_error_hints import structured_tool_error
from tools.sql_guard import query_with_columns_timed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualization"


def _to_float(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return 0.0
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def create_chart_from_sql(
    db: DuckDBQuery,
    sql: str,
    chart_type: str = "bar",
    title: str | None = None,
    output_dir: Path | None = None,
) -> dict:
    """
    Run read-only SQL that returns >=2 columns: first = category/date label, second = numeric measure.
    Saves PNG under outputs/visualization/ and returns paths + markdown embed snippet.
    """
    chart_type = (chart_type or "bar").lower().strip()
    if chart_type not in ("bar", "line"):
        return structured_tool_error(
            f"chart_type must be 'bar' or 'line', got {chart_type!r}",
            error_kind="chart_bad_type",
            next_step="Set chart_type to 'bar' or 'line'.",
        )

    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    columns, rows, err_json = query_with_columns_timed(db, sql)
    if err_json:
        return json.loads(err_json)
    if not rows:
        return structured_tool_error(
            "Query returned no rows — nothing to chart.",
            error_kind="chart_empty_rows",
            next_step="Loosen filters or widen the cohort so the SQL returns rows, then retry create_chart.",
        )
    if len(columns) < 2:
        return structured_tool_error(
            "Need at least two columns: first = labels (category or date), second = numeric values.",
            error_kind="chart_bad_columns",
            next_step="SELECT at least two columns (label, numeric measure) with LIMIT on SELECT...FROM.",
        )

    max_points = 40
    rows = rows[:max_points]

    labels = [str(r[0])[:100] if r[0] is not None else "" for r in rows]
    values = [_to_float(r[1]) for r in rows]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(labels))

    col_x = str(columns[0])
    col_y = str(columns[1])
    heading = (title or "").strip() or f"{col_y} by {col_x}"

    if chart_type == "bar":
        ax.bar(x, values, color="steelblue", edgecolor="white", linewidth=0.5)
    else:
        ax.plot(x, values, marker="o", color="steelblue", linewidth=2, markersize=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    # Heading (chart title) + explicit axis labels from SQL column names
    ax.set_title(heading, fontsize=14, fontweight="bold", pad=18)
    ax.set_xlabel(col_x, fontsize=11, labelpad=10)
    ax.set_ylabel(col_y, fontsize=11, labelpad=10)

    fig.tight_layout(pad=1.2)

    fname = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    rel = out_path.relative_to(PROJECT_ROOT)
    rel_str = rel.as_posix()
    ttl = heading

    sample_rows = [
        {"label": labels[i], "value": values[i]}
        for i in range(min(12, len(labels)))
    ]

    return {
        "chart_path": rel_str,
        "title": ttl,
        "chart_type": chart_type,
        "rows_plotted": len(rows),
        "columns": list(columns[:2]),
        "sample_rows": sample_rows,
        "markdown_embed": f"![{ttl}]({rel_str})",
        "note": "Include markdown_embed in the final answer so the user can see where the PNG lives; session export will contain this JSON.",
    }
