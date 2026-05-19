# Demo workflow: diabetes cohort + LAB care-gap analysis

Use this as a **single end-to-end story** for screenshots or live demos. It combines **cohort definition** (E11 diagnosis proxy + metformin exposure) with a **care-gap style question** (LAB visit utilization on medical claims). Data is **synthetic**—not for clinical, quality, or payment decisions.

## What you are showing

| Step | Analytics idea | App features exercised |
|------|----------------|-------------------------|
| 1 | Define a coherent patient cohort | `query_database`, planner outline, suggested follow-ups |
| 2 | Quantify a gap (no LAB visit type among cohort) | Multi-step SQL, interpretation in prose |
| 3 | (Optional) Chart a trend for the same cohort | `create_chart`, PNG + markdown embed |
| 4 | Follow-up in plain English | **Cohort memory** (“same cohort…”) if you reference prior SQL |

## Natural-language questions (copy into the chat)

**Cohort size**

- “How many distinct patients have at least one **E11%** admit diagnosis on `mx_events` and at least one **metformin** fill on `rx_events`?”

**Care gap (utilization proxy)**

- “Among those patients, how many **never** have a medical claim row with `VISIT_TYPE = 'LAB'`? Give me the count of cohort patients and the count without any LAB visit type.”

**Trend (optional chart)**

- “For that same diabetes-and-metformin cohort, chart **monthly metformin fills** over time—two columns for `create_chart`, line chart, title that names the cohort and metric.”

**Continuity**

- “Using the **same cohort** as before, stratify the LAB-gap summary by **patient sex** from demographics.”

## Reference SQL (agent may vary wording; columns must match your DB)

Cohort count (E11 + metformin):

```sql
WITH dm_patients AS (
  SELECT DISTINCT "PATIENT_NUMBER"
  FROM mx_events
  WHERE "ADMIT_DIAGNOSIS_CODE" LIKE 'E11%'
),
metf AS (
  SELECT DISTINCT "PATIENT_NUMBER"
  FROM rx_events
  WHERE LOWER(COALESCE("GENERIC_NAME", '')) LIKE '%metformin%'
)
SELECT COUNT(DISTINCT m."PATIENT_NUMBER") AS patients_dm_and_metformin
FROM metf AS m
INNER JOIN dm_patients AS d ON m."PATIENT_NUMBER" = d."PATIENT_NUMBER"
LIMIT 1;
```

LAB gap among E11 patients (from [`diabetes_workflow_sql_examples.md`](diabetes_workflow_sql_examples.md) section 3):

```sql
WITH dm AS (
  SELECT DISTINCT "PATIENT_NUMBER"
  FROM mx_events
  WHERE "ADMIT_DIAGNOSIS_CODE" LIKE 'E11%'
),
lab AS (
  SELECT DISTINCT "PATIENT_NUMBER"
  FROM mx_events
  WHERE "VISIT_TYPE" = 'LAB'
)
SELECT
  COUNT(*) AS dm_patients,
  SUM(CASE WHEN lab."PATIENT_NUMBER" IS NULL THEN 1 ELSE 0 END) AS dm_without_lab_visit_type
FROM dm
LEFT JOIN lab ON dm."PATIENT_NUMBER" = lab."PATIENT_NUMBER"
LIMIT 1;
```

## Screenshot tips

1. Show **planner** expander (Streamlit) or `[Planner]` block (terminal) + tool trace + final answer.  
2. Show **Reviewer Evaluation** expander with peer review markdown.  
3. Export session markdown once and mention **Executive summary** + reproducibility block.

See also: [`diabetes_workflow_sql_examples.md`](diabetes_workflow_sql_examples.md), [`README.md`](../README.md) (architecture), [`ARCHITECTURE.md`](../ARCHITECTURE.md).
