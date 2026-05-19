# Cost aggregates and charts (`create_chart`)

Synthetic data only — not for clinical or payment operations.

## Cost-style aggregates (medical claims)

`mx_events` includes billing-style fields such as `PROVIDER_BILLED`, `PLAN_ALLOWED`, and `PLAN_PAID` (see `describe_table` / `profile_table` for exact types).

**How to read these numbers:** Sums and rankings here are **exploratory** on synthetic claims. Higher `PROVIDER_BILLED` or RX spend does **not** imply worse or better care—differences can reflect **volume**, **case mix**, **coding**, geography, or plan design. There is **no** risk adjustment, episode grouping, or normalized price in this starter schema. Use aggregates for directional patterns only, not payment or quality decisions.

**Overall totals:**

```sql
SELECT
  SUM("PROVIDER_BILLED") AS total_provider_billed,
  SUM("PLAN_ALLOWED")    AS total_plan_allowed,
  SUM("PLAN_PAID")       AS total_plan_paid
FROM mx_events
LIMIT 1;
```

**By state (top 15 by billed amount):**

```sql
SELECT
  "PATIENT_STATE" AS state,
  SUM("PROVIDER_BILLED") AS total_billed
FROM mx_events
GROUP BY 1
ORDER BY total_billed DESC
LIMIT 15;
```

Use this pattern with `query_database`, or feed the same two-column shape to **`create_chart`** (`chart_type`: `bar`).

## Pharmacy spend (RX)

Example:

```sql
SELECT
  SUM("PHARMACY_SUBMITTED_AMOUNT") AS total_submitted,
  SUM("PLAN_PAID") AS total_plan_paid
FROM rx_events
LIMIT 1;
```

## Chart tool (`create_chart`)

Chart SQL uses the **same guardrails as `query_database`**: `SELECT ... FROM ...` must include **`LIMIT n`**, and queries are subject to the shared **timeout** ([`tools/sql_guard.py`](../tools/sql_guard.py)).

The agent calls **`create_chart`** with:

- **`sql`**: `SELECT` returning **exactly two columns** — label, then numeric value (add `LIMIT` / `GROUP BY` as needed).
- **`chart_type`**: `bar` or `line`.
- **`title`**: Short figure title that states **what** is plotted: population or cohort (when relevant), **metric**, and **time span or breakdown** (e.g. `Metformin fills by month, E11 cohort, study window`).

The tool writes `outputs/visualization/chart_<timestamp>.png` and returns **`markdown_embed`**. The assistant should repeat that line in the chat answer; session markdown export includes the tool JSON for audit.

**Example chart query (monthly volume — adjust date function if needed):**

```sql
SELECT
  CAST(DATE_TRUNC('month', CAST("SERVICE_DATE" AS DATE)) AS VARCHAR) AS month_start,
  COUNT(*) AS event_count
FROM mx_events
GROUP BY 1
ORDER BY 1
LIMIT 24;
```

Then `create_chart` with `chart_type: line` and `title`, e.g. `MX event volume by calendar month, all mx_events`.
