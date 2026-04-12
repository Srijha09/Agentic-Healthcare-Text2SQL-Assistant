# Plan: Synthetic Healthcare Data Generator

## Overview
Python script (`generate_data.py`) that generates synthetic healthcare data for an interview SQL agent assessment. The data will span 5 tables with a shared `PATIENT_NUMBER` key enabling meaningful joins.

## Output Location
- **Script**: `data/generate_data.py`
- **Output CSVs**: `data/`

## Reference Data Sources (in `data/input.zip`)
- `icd10_codes.csv` - Diagnosis codes (CODE, DESCRIPTION)
- `procedure_codes.csv` - CPT/HCPCS procedure codes (CODE, DESCRIPTION)
- `ndc_products.csv` - NDC11 codes, drug names, routes

---

## Configurable Global Variables

```python
# Patient counts
NUM_PATIENTS = 10_000
MORTALITY_RATE = 0.05          # 5% of patients have death records

# Events per patient (uniform random in range)
MX_EVENTS_MIN = 1
MX_EVENTS_MAX = 500
RX_EVENTS_MIN = 1
RX_EVENTS_MAX = 500

# Date range
DATE_START = "2020-01-01"
DATE_END = "2025-12-31"

# Diagnosis codes per MX event
DIAGNOSIS_CODES_MIN = 1
DIAGNOSIS_CODES_MAX = 50

# Random seed for reproducibility
RANDOM_SEED = 69_420
```

---

## Generation Pipeline

### Step 1: Generate Patient IDs

**Column: `PATIENT_NUMBER`**
- Generate `NUM_PATIENTS` unique integers
- Range: 9-10 digit integers (100,000,000 to 9,999,999,999)
- Method: `np.random.randint()` then dedupe

Output: `patient_ids` array of 10K unique integers

---

### Step 1b: Generate Patient-Level Attributes (not a table, used for populating other tables)

Generate a dictionary/DataFrame mapping each patient to attributes that need to be consistent across tables:

| Attribute | Generation Method |
|-----------|-------------------|
| `PATIENT_NUMBER` | From Step 1 |
| `INSURANCE_GROUP` | Random from `['MEDICARE', 'MEDICAID', 'COMMERCIAL', 'VA', None]` |
| `PATIENT_YOB_DATE` | Random date, year only (Jan 1 of random year 1930-2020) |
| `PATIENT_SEX` | Random choice from `['M', 'F', 'U']` with weights `[0.48, 0.48, 0.04]` |
| `PATIENT_STATE` | Random choice from 50 US state codes + DC |
| `PATIENT_ZIP3` | Random 3-digit string, zero-padded |

This patient attributes lookup is used when generating demographics, geography, MX events, and RX events to ensure consistency.

---

### Step 2: Generate Demographics Table

**One row per patient.**

| Column | Generation Method |
|--------|-------------------|
| `PATIENT_NUMBER` | From Step 1 |
| `PATIENT_YOB_DATE` | Random date, year only (Jan 1 of random year 1930-2020), ensures age 5-95 |
| `PATIENT_SEX` | Random choice from `['M', 'F', 'U']` with weights `[0.48, 0.48, 0.04]` |

---

### Step 3: Generate Geography Table

**One row per patient.**

| Column | Generation Method |
|--------|-------------------|
| `PATIENT_NUMBER` | From Step 1 |
| `PATIENT_STATE` | Random choice from list of 50 US state codes + DC |
| `PATIENT_ZIP` | Random 3-digit string, zero-padded (`'001'` to `'999'`) |
| `VALID_FROM_DATE` | Random date in range, before `VALID_TO_DATE` |
| `VALID_TO_DATE` | Random date after `VALID_FROM_DATE`, or `None` (still valid) for 50% |

---

### Step 4: Generate Mortality Table

**~5% of patients (500 patients).**

| Column | Generation Method |
|--------|-------------------|
| `PATIENT_NUMBER` | Random sample of `NUM_PATIENTS * MORTALITY_RATE` patient IDs |
| `PATIENT_DEATH_DATE` | Random date in `DATE_START` to `DATE_END`, day set to 1st of month |

---

### Step 5: Generate MX Events Table

**1-500 events per patient (configurable).**

For each patient:
1. Sample `num_events = randint(MX_EVENTS_MIN, MX_EVENTS_MAX)`
2. Generate `num_events` rows

| Column | Generation Method |
|--------|-------------------|
| `CLAIM_UUID` | Random 12-digit integer |
| `PATIENT_NUMBER` | Current patient ID |
| `SERVICE_DATE` | Random date in `DATE_START` to `DATE_END` |
| `SERVICE_TO_DATE` | `SERVICE_DATE` + random 0-30 days |
| `PROCEDURE_CODE` | Sample from `komodo_procedures.csv` CODE column |
| `PROCEDURE_CODE_TYPE` | Random from `['CPT', 'HCPCS', 'ICD-10-PCS']` |
| `NDC11` | Sample from `ndc_package.csv` NDC11 column (nullable, 30% populated) |
| `UNITS` | Random int 1-100 |
| `UNIT_TYPE` | Random from schema samples: `['1 EA', '10 ML', '1 MG', ...]` |
| `RENDERING_NPI` | Random 10-digit starting with '1' |
| `REFERRING_NPI` | Random 10-digit starting with '1' |
| `BILLING_NPI` | Random 10-digit starting with '1' |
| `FACILITY_NPI` | Random 10-digit starting with '1' |
| `VISIT_TYPE` | Random from `['OUTPATIENT', 'INPATIENT', 'EMERGENCY', 'TELEHEALTH', 'LAB', 'HOME', 'ASC', 'OTHER']` |
| `INSURANCE_GROUP` | From patient attributes lookup |
| `DIAGNOSIS_CODES` | Sample `DIAGNOSIS_CODES_MIN` to `DIAGNOSIS_CODES_MAX` codes from `icd10_codes.csv`, pipe-delimited: `\|A01\|B02\|` |
| `ADMIT_DIAGNOSIS_CODE` | Single code from `icd10_codes.csv` (nullable, 40% populated) |
| `EVENT_SOURCE` | Random from `['INSTITUTIONAL', 'PROFESSIONAL']` |
| `PROVIDER_BILLED` | Random float 10.00-50000.00 |
| `PLAN_ALLOWED` | Random float, <= `PROVIDER_BILLED` |
| `PLAN_PAID` | Random float, <= `PLAN_ALLOWED` |
| `PATIENT_YOB` | From patient attributes lookup |
| `PATIENT_SEX` | From patient attributes lookup |
| `PATIENT_ZIP3` | From patient attributes lookup |
| `PATIENT_STATE` | From patient attributes lookup |

---

### Step 6: Generate RX Events Table

**1-500 events per patient (configurable).**

For each patient:
1. Sample `num_events = randint(RX_EVENTS_MIN, RX_EVENTS_MAX)`
2. Generate `num_events` rows

| Column | Generation Method |
|--------|-------------------|
| `CLAIM_UUID` | Random 12-digit integer |
| `PATIENT_NUMBER` | Current patient ID |
| `FILL_DATE` | Random date in `DATE_START` to `DATE_END` |
| `PHARMACY_NPI` | Random 10-digit starting with '1' |
| `PHARMACY_CHANNEL` | Random from `['RETAIL', 'MAIL', 'SPECIALTY', 'LTC', 'PROVIDER', 'UNKNOWN']` |
| `PRESCRIBER_NPI` | Random 10-digit starting with '1' |
| `NDC11` | Sample a row from `ndc_product.csv`, take NDC11 |
| `BRAND_NAME` | From same sampled row, take PROPRIETARYNAME |
| `GENERIC_NAME` | From same sampled row, take NONPROPRIETARYNAME |
| `ROUTE` | From same sampled row, take ROUTENAME |
| `DAYS_SUPPLY` | Random int 1-90, with some outliers up to 365 |
| `QUANTITY` | Random float 1-500 |
| `DIAGNOSIS_CODE` | Single code from `icd10_codes.csv` |
| `TRANSACTION_RESULT` | Random from `['PAID', 'REVERSED', 'REJECTED', 'PENDED']` |
| `TRANSACTION_NUMBER` | Random int 1-100 |
| `TRANSACTION_STATUS` | Random from `['INITIAL', 'FINAL', 'INTERIM', 'STANDALONE']` |
| `FILL_NUMBER` | Random int 0-99 |
| `NUMBER_OF_REFILLS_AUTHORIZED` | Random int 0-99 |
| `DATE_PRESCRIPTION_WRITTEN` | Random date, on or before `FILL_DATE` |
| `INSURANCE_GROUP` | From patient attributes lookup |
| `PHARMACY_SUBMITTED_AMOUNT` | Random float 10.00-5000.00 |
| `PLAN_PAID` | Random float <= submitted amount |
| `PATIENT_RESPONSIBILITY` | Random float 0-500 |
| `PATIENT_OOP` | Same or similar to `PATIENT_RESPONSIBILITY` |
| `PATIENT_YOB` | From patient attributes lookup |
| `PATIENT_SEX` | From patient attributes lookup |
| `PATIENT_ZIP3` | From patient attributes lookup |
| `PATIENT_STATE` | From patient attributes lookup |

---

## Script Structure

```
1. Imports & Config
   - pandas, numpy, random, tqdm
   - Global config variables

2. Load Reference Data (from input.zip)
   - Load ICD10 codes
   - Load procedure codes
   - Load NDC products

3. Helper Functions
   - generate_npi() -> 10-digit NPI string starting with '1'
   - generate_patient_ids(n) -> array of unique IDs
   - random_date(start, end) -> random date in range
   - sample_diagnosis_codes(n) -> pipe-delimited string

4. Generate Patient Base Data
   - patient_ids array
   - demographics_df
   - geography_df

5. Generate Subset Tables
   - mortality_df (5% of patients)

6. Generate Event Tables
   - mx_events_df (vectorized, 1-500 events per patient)
   - rx_events_df (vectorized, 1-500 events per patient)

7. Export to CSV
   - Save all DataFrames to data/
```

---

## Output Files

All CSVs written to `data/`:

**Patient Data (generated):**
- `demographics.csv`
- `geography.csv`
- `mortality.csv`
- `mx_events.csv`
- `rx_events.csv`

**Code Lookup Tables (from input.zip):**
- `icd10_codes.csv` - columns: `CODE`, `DESCRIPTION`
- `procedure_codes.csv` - columns: `CODE`, `DESCRIPTION`
- `ndc_products.csv` - NDC11 codes with drug info

---

## Excluded Columns (Internal/Meta - NOT generated)

These columns from the original schemas will NOT be included:
- `KH_META_*` (all meta timestamp/hash columns)
- `*_PARTITION_KEY*`
- `*_BITSET` columns
- `SOURCE_ALIASES` / `ROOT_SOURCE_ALIASES*` arrays
- `*_UNCERTIFIED` date variants (keeping only main date columns)
- `MIN_*` / `MAX_*` aggregate columns from demographics
