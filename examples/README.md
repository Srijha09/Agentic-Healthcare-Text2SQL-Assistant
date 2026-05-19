# Example SQL and prompts

This folder holds **reference SQL and copy-paste prompts** for the synthetic healthcare DuckDB used by the chat app. Nothing here runs automatically; use these with `query_database` / `create_chart` in the agent, or run in DuckDB after you confirm column names (`DESCRIBE table_name`).

| File | What it is |
|------|------------|
| [`demo_cohort_and_care_gap_workflow.md`](demo_cohort_and_care_gap_workflow.md) | **Recommended for demos:** one coherent story—diabetes/metformin **cohort** + **LAB care-gap** questions, optional chart, cohort-memory follow-ups. |
| [`diabetes_workflow_sql_examples.md`](diabetes_workflow_sql_examples.md) | Full SQL snippets: E11 proxy, metformin, trends, LAB gap, provider billed + sample prompts. |
| [`time_to_metformin_sql_examples.md`](time_to_metformin_sql_examples.md) | Time from first diagnosis-line signal to first metformin fill (day-diff patterns). |
| [`cost_and_chart_sql_examples.md`](cost_and_chart_sql_examples.md) | `PROVIDER_BILLED` / plan fields, RX spend aggregates, and how `create_chart` expects two-column SQL. |
| [`cohort_trends_note.md`](cohort_trends_note.md) | Short pointer to the main diabetes workflow doc for cohort/trend SQL. |

All examples assume **synthetic claims-style data** — not for clinical, payment, or regulatory decisions.
