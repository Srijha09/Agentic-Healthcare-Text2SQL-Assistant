# Time-to-event patterns (diagnosis signal to first metformin)

Synthetic claims data only — not for clinical or payment decisions.

These queries use the same **E11 admit proxy** and **metformin** match as [`workflow_flagship_diabetes_analytics.md`](workflow_flagship_diabetes_analytics.md). Confirm column names with `DESCRIBE mx_events` / `DESCRIBE rx_events`.

## Days from first E11 admit-line signal to first metformin fill (cohort-level)

Restricts to patients where the first metformin fill is **on or after** the first qualifying MX service date (otherwise the gap is not meaningful for “treatment after dx signal”).

```sql
WITH dm_first AS (
  SELECT "PATIENT_NUMBER", MIN(CAST("SERVICE_DATE" AS DATE)) AS first_dm_date
  FROM mx_events
  WHERE "ADMIT_DIAGNOSIS_CODE" LIKE 'E11%'
  GROUP BY 1
),
met_first AS (
  SELECT "PATIENT_NUMBER", MIN(CAST("FILL_DATE" AS DATE)) AS first_met_date
  FROM rx_events
  WHERE LOWER(COALESCE("GENERIC_NAME", '')) LIKE '%metformin%'
  GROUP BY 1
)
SELECT
  AVG(date_diff('day', d.first_dm_date, m.first_met_date)) AS avg_days,
  median(date_diff('day', d.first_dm_date, m.first_met_date)) AS median_days,
  COUNT(*) AS patient_count
FROM dm_first AS d
INNER JOIN met_first AS m ON d."PATIENT_NUMBER" = m."PATIENT_NUMBER"
WHERE m.first_met_date >= d.first_dm_date
LIMIT 1;
```

## Stratification idea (by state)

Join `demographics` / `geography` on `PATIENT_NUMBER` and repeat the per-patient day-diff in a subquery, then `GROUP BY` state with a small `LIMIT` on the number of states returned. Use this as a template after verifying join keys in your load.

## Natural-language prompts

- “Among patients with an E11 admit code and a metformin fill after their first E11 service date, what is the median days from first E11 MX date to first metformin fill?”
- “Chart average days from first E11 to first metformin by patient state (top 10 states by cohort size).”
