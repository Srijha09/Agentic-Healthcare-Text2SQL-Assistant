# Diabetes cohort workflow — SQL examples (cohort, therapy, gaps, provider cost)

This is the **primary healthcare narrative** for demos: concrete use cases aligned with analytics on claims-like data (synthetic only — not for clinical or payment decisions).

Use it end-to-end with the chat app: the **Planner** step outlines tables and joins; **execution** runs `query_database` / `create_chart`; **export** (`export` or exit) writes markdown with **canonical SQL** and tool logs.

## Definitions and limitations

- **Diabetes cohort signal:** `ADMIT_DIAGNOSIS_CODE LIKE 'E11%'` on **mx_events** is a **proxy** for type 2 diabetes documentation on a claim line. It does **not** represent a full problem list, all claim-line diagnoses, or clinical certainty.
- **Metformin exposure:** Matched via `rx_events.GENERIC_NAME` (or NDC paths in other analyses). Indicates at least one fill in the synthetic history, not adherence or dose optimization.
- **LAB “care gap”:** `VISIT_TYPE = 'LAB'` is a **utilization proxy**, not a specific HbA1c or LOINC result (this schema does not model individual lab results).
- **Provider cost:** `PROVIDER_BILLED` sums are **claims dollars in synthetic data**, not quality scores, standardized costs, or case-mix–adjusted comparisons.
- **Agent SQL:** The app **requires `LIMIT n`** on `SELECT ... FROM ...` queries. Add `LIMIT 1` for single-row aggregates and a bounded `LIMIT` for time series or lists.

## 1) Cohort: type 2 diabetes signal + metformin exposure

**Story:** Identify patients with a diabetes diagnosis signal on MX and at least one metformin fill on RX.

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

## 2) Trend: metformin fills over time (therapy persistence proxy)

**Story:** Monthly count of metformin fills in the diabetes-admit slice — useful with `create_chart` (two columns: month label, count).

```sql
WITH dm AS (
  SELECT DISTINCT "PATIENT_NUMBER"
  FROM mx_events
  WHERE "ADMIT_DIAGNOSIS_CODE" LIKE 'E11%'
)
SELECT
  DATE_TRUNC('month', CAST(r."FILL_DATE" AS DATE)) AS month_start,
  COUNT(*) AS metformin_fills
FROM rx_events AS r
INNER JOIN dm ON r."PATIENT_NUMBER" = dm."PATIENT_NUMBER"
WHERE LOWER(COALESCE(r."GENERIC_NAME", '')) LIKE '%metformin%'
GROUP BY 1
ORDER BY 1
LIMIT 500;
```

## 3) Care gap proxy: diabetic cohort without a LAB visit

**Story:** Among patients with an E11 admit diagnosis, how many never have an MX row with `VISIT_TYPE = 'LAB'` (follow-up lab utilization proxy — not a specific LOINC test in this schema).

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

## 4) Provider-level cost: high-risk slice (diabetes + metformin)

**Story:** Rank rendering providers by total `PROVIDER_BILLED` among patients who are in both the diabetes admit cohort and metformin users.

```sql
WITH dm AS (
  SELECT DISTINCT "PATIENT_NUMBER"
  FROM mx_events
  WHERE "ADMIT_DIAGNOSIS_CODE" LIKE 'E11%'
),
metf AS (
  SELECT DISTINCT "PATIENT_NUMBER"
  FROM rx_events
  WHERE LOWER(COALESCE("GENERIC_NAME", '')) LIKE '%metformin%'
),
cohort AS (
  SELECT d."PATIENT_NUMBER"
  FROM dm AS d
  INNER JOIN metf AS m ON d."PATIENT_NUMBER" = m."PATIENT_NUMBER"
)
SELECT
  mx."RENDERING_NPI",
  SUM(mx."PROVIDER_BILLED") AS total_billed,
  COUNT(*) AS mx_rows
FROM mx_events AS mx
INNER JOIN cohort AS c ON mx."PATIENT_NUMBER" = c."PATIENT_NUMBER"
GROUP BY 1
ORDER BY total_billed DESC
LIMIT 25;
```

## Example prompts for the agent

- “How many patients have an E11 admit diagnosis and at least one metformin fill? Then chart metformin fills per month for that cohort.”
- “What share of E11 patients never have a LAB visit type on MX?”
- “Top 10 rendering NPIs by total PROVIDER_BILLED for patients on metformin with a diabetes admit code.”

See also [`cost_and_chart_sql_examples.md`](cost_and_chart_sql_examples.md) for cost fields and chart usage. **Time from diagnosis signal to first metformin:** [`time_to_metformin_sql_examples.md`](time_to_metformin_sql_examples.md).
