#!/usr/bin/env python3
"""
Synthetic Healthcare Data Generator

Generates synthetic healthcare data for interview SQL agent assessment.
"""

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# CONFIGURABLE GLOBAL VARIABLES
# =============================================================================

# Patient counts
NUM_PATIENTS = 10_000
MORTALITY_RATE = 0.05  # 5% of patients have death records

# Events per patient (uniform random in range)
MX_EVENTS_MIN = 1
MX_EVENTS_MAX = 500
RX_EVENTS_MIN = 1
RX_EVENTS_MAX = 500

# Date range
DATE_START = date(2020, 1, 1)
DATE_END = date(2025, 12, 31)

# Diagnosis codes per MX event
DIAGNOSIS_CODES_MIN = 1
DIAGNOSIS_CODES_MAX = 50

# NPI pool sizes (for joinability across events)
NUM_RENDERING_NPIS = 10_000
NUM_REFERRING_NPIS = 8_000
NUM_BILLING_NPIS = 3_000
NUM_FACILITY_NPIS = 1_500
NUM_PHARMACY_NPIS = 2_000
NUM_PRESCRIBER_NPIS = 10_000

# Random seed for reproducibility
RANDOM_SEED = 69_420

# US State codes
US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
]

# Value lists for random sampling
SEX_VALUES = ["M", "F", "U"]
SEX_WEIGHTS = [0.48, 0.48, 0.04]
INSURANCE_GROUPS = ["MEDICARE", "MEDICAID", "COMMERCIAL", "VA", None]
VISIT_TYPES = ["OUTPATIENT", "INPATIENT", "EMERGENCY", "TELEHEALTH", "LAB", "HOME", "ASC", "OTHER"]
EVENT_SOURCES = ["INSTITUTIONAL", "PROFESSIONAL"]
PROCEDURE_CODE_TYPES = ["CPT", "HCPCS", "ICD-10-PCS"]
UNIT_TYPES = ["1 EA", "10 ML", "1 MG", "100 MG", "1 GM", "1 TABLET", "1 CAPSULE", "1 VIAL"]
PHARMACY_CHANNELS = ["RETAIL", "MAIL", "SPECIALTY", "LTC", "PROVIDER", "UNKNOWN"]
TRANSACTION_RESULTS = ["PAID", "REVERSED", "REJECTED", "PENDED"]
TRANSACTION_STATUSES = ["INITIAL", "FINAL", "INTERIM", "STANDALONE"]


def generate_npi() -> str:
    """Generate a random 10-digit NPI starting with '1'."""
    return "1" + "".join([str(random.randint(0, 9)) for _ in range(9)])


def generate_npi_pool(n: int) -> list:
    """Generate a pool of n unique NPIs."""
    npis = set()
    while len(npis) < n:
        npis.add(generate_npi())
    return list(npis)


def generate_patient_ids(n: int) -> np.ndarray:
    """Generate n unique patient IDs (9-10 digit integers)."""
    ids = set()
    while len(ids) < n:
        new_ids = np.random.randint(100_000_000, 10_000_000_000, size=n - len(ids), dtype=np.int64)
        ids.update(new_ids.tolist())
    return np.array(list(ids)[:n])


def random_date(start: date, end: date) -> date:
    """Generate a random date between start and end."""
    delta = (end - start).days
    random_days = random.randint(0, delta)
    return start + timedelta(days=random_days)


def random_dates_array(start: date, end: date, size: int) -> list:
    """Generate an array of random dates."""
    delta = (end - start).days
    random_days = np.random.randint(0, delta + 1, size=size)
    return [start + timedelta(days=int(d)) for d in random_days]


def main():
    # Set random seeds
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Paths - script is in data/ directory, outputs go to same directory
    DATA_DIR = Path(__file__).parent
    print(f"Data directory: {DATA_DIR}")

    # =========================================================================
    # Load Reference Data
    # =========================================================================
    print("\nLoading reference data...")

    icd10_df = pd.read_csv(DATA_DIR / "icd10_codes.csv")
    icd10_codes = icd10_df["CODE"].tolist()
    print(f"  Loaded {len(icd10_codes):,} ICD10 codes")

    procedures_df = pd.read_csv(DATA_DIR / "procedure_codes.csv")
    procedure_codes = procedures_df["CODE"].tolist()
    print(f"  Loaded {len(procedure_codes):,} procedure codes")

    ndc_product_df = pd.read_csv(DATA_DIR / "ndc_products.csv")
    ndc_product_df = ndc_product_df[ndc_product_df["NDC11"].notna()].copy()
    ndc_product_df["NDC11"] = ndc_product_df["NDC11"].astype(str).str.zfill(11)
    print(f"  Loaded {len(ndc_product_df):,} NDC products")

    def sample_diagnosis_codes(n: int) -> str:
        """Sample n diagnosis codes and return as pipe-delimited string."""
        codes = random.sample(icd10_codes, min(n, len(icd10_codes)))
        return "|" + "|".join(codes) + "|"

    # =========================================================================
    # Generate NPI Pools
    # =========================================================================
    print("\nGenerating NPI pools...")
    RENDERING_NPI_POOL = generate_npi_pool(NUM_RENDERING_NPIS)
    REFERRING_NPI_POOL = generate_npi_pool(NUM_REFERRING_NPIS)
    BILLING_NPI_POOL = generate_npi_pool(NUM_BILLING_NPIS)
    FACILITY_NPI_POOL = generate_npi_pool(NUM_FACILITY_NPIS)
    PHARMACY_NPI_POOL = generate_npi_pool(NUM_PHARMACY_NPIS)
    PRESCRIBER_NPI_POOL = generate_npi_pool(NUM_PRESCRIBER_NPIS)
    print("  Done generating NPI pools")

    # =========================================================================
    # Generate Patient IDs & Attributes
    # =========================================================================
    print("\nGenerating patient data...")
    patient_ids = generate_patient_ids(NUM_PATIENTS)
    print(f"  Generated {len(patient_ids):,} unique patient IDs")

    patient_attributes = pd.DataFrame({
        "PATIENT_NUMBER": patient_ids,
        "PATIENT_YOB_DATE": [date(random.randint(1930, 2020), 1, 1) for _ in range(NUM_PATIENTS)],
        "PATIENT_SEX": np.random.choice(SEX_VALUES, size=NUM_PATIENTS, p=SEX_WEIGHTS),
        "PATIENT_STATE": np.random.choice(US_STATES, size=NUM_PATIENTS),
        "PATIENT_ZIP3": [str(random.randint(1, 999)).zfill(3) for _ in range(NUM_PATIENTS)],
        "INSURANCE_GROUP": np.random.choice(INSURANCE_GROUPS, size=NUM_PATIENTS),
    })
    patient_lookup = patient_attributes.set_index("PATIENT_NUMBER").to_dict("index")

    # =========================================================================
    # Generate Demographics Table
    # =========================================================================
    demographics_df = pd.DataFrame({
        "PATIENT_NUMBER": patient_attributes["PATIENT_NUMBER"],
        "PATIENT_YOB_DATE": patient_attributes["PATIENT_YOB_DATE"],
        "PATIENT_SEX": patient_attributes["PATIENT_SEX"],
    })
    print(f"  Generated demographics table: {len(demographics_df):,} rows")

    # =========================================================================
    # Generate Geography Table
    # =========================================================================
    valid_from_dates = random_dates_array(DATE_START, DATE_END, NUM_PATIENTS)
    valid_to_dates = []
    for from_date in valid_from_dates:
        if random.random() < 0.5:
            valid_to_dates.append(None)
        else:
            days_remaining = (DATE_END - from_date).days
            if days_remaining > 0:
                valid_to_dates.append(from_date + timedelta(days=random.randint(1, days_remaining)))
            else:
                valid_to_dates.append(None)

    geography_df = pd.DataFrame({
        "PATIENT_NUMBER": patient_attributes["PATIENT_NUMBER"],
        "PATIENT_STATE": patient_attributes["PATIENT_STATE"],
        "PATIENT_ZIP": patient_attributes["PATIENT_ZIP3"],
        "VALID_FROM_DATE": valid_from_dates,
        "VALID_TO_DATE": valid_to_dates,
    })
    print(f"  Generated geography table: {len(geography_df):,} rows")

    # =========================================================================
    # Generate Mortality Table
    # =========================================================================
    num_deceased = int(NUM_PATIENTS * MORTALITY_RATE)
    deceased_patient_ids = np.random.choice(patient_ids, size=num_deceased, replace=False)
    death_dates = [date(random_date(DATE_START, DATE_END).year, random_date(DATE_START, DATE_END).month, 1) for _ in range(num_deceased)]

    mortality_df = pd.DataFrame({
        "PATIENT_NUMBER": deceased_patient_ids,
        "PATIENT_DEATH_DATE": death_dates,
    })
    print(f"  Generated mortality table: {len(mortality_df):,} rows")

    # =========================================================================
    # Generate MX Events Table
    # =========================================================================
    print("\nGenerating MX events...")
    patient_event_counts = np.random.randint(MX_EVENTS_MIN, MX_EVENTS_MAX + 1, size=NUM_PATIENTS)
    total_mx_events = patient_event_counts.sum()
    print(f"  Total MX events: {total_mx_events:,}")

    patient_numbers = np.repeat(patient_ids, patient_event_counts)
    service_dates = random_dates_array(DATE_START, DATE_END, total_mx_events)
    service_durations = np.random.randint(0, 31, size=total_mx_events)
    service_to_dates = [min(sd + timedelta(days=int(dur)), DATE_END) for sd, dur in zip(service_dates, service_durations)]

    rendering_npis = np.random.choice(RENDERING_NPI_POOL, size=total_mx_events)
    referring_npis = np.random.choice(REFERRING_NPI_POOL, size=total_mx_events)
    billing_npis = np.random.choice(BILLING_NPI_POOL, size=total_mx_events)
    facility_npis = np.random.choice(FACILITY_NPI_POOL, size=total_mx_events)

    procedure_codes_arr = np.random.choice(procedure_codes, size=total_mx_events)
    procedure_code_types = np.random.choice(PROCEDURE_CODE_TYPES, size=total_mx_events)
    units = np.random.randint(1, 101, size=total_mx_events)
    unit_types = np.random.choice(UNIT_TYPES, size=total_mx_events)
    visit_types = np.random.choice(VISIT_TYPES, size=total_mx_events)
    event_sources = np.random.choice(EVENT_SOURCES, size=total_mx_events)

    provider_billed = np.round(np.random.uniform(10.0, 50000.0, size=total_mx_events), 2)
    plan_allowed = np.round(np.random.uniform(10.0, 1.0, size=total_mx_events) * provider_billed, 2)
    plan_paid = np.round(np.random.uniform(0.0, 1.0, size=total_mx_events) * plan_allowed, 2)

    claim_uuids = np.random.randint(100_000_000_000, 999_999_999_999, size=total_mx_events, dtype=np.int64)

    ndc_mask = np.random.random(size=total_mx_events) < 0.3
    ndc_indices = np.random.randint(0, len(ndc_product_df), size=total_mx_events)
    ndc11_values = np.where(ndc_mask, ndc_product_df.iloc[ndc_indices]['NDC11'].values, None)

    admit_mask = np.random.random(size=total_mx_events) < 0.4
    admit_dx_indices = np.random.randint(0, len(icd10_codes), size=total_mx_events)
    admit_dx_values = np.where(admit_mask, np.array(icd10_codes)[admit_dx_indices], None)

    insurance_groups = np.array([patient_lookup[pid]['INSURANCE_GROUP'] for pid in patient_numbers])
    patient_yobs = np.array([patient_lookup[pid]['PATIENT_YOB_DATE'] for pid in patient_numbers])
    patient_sexes = np.array([patient_lookup[pid]['PATIENT_SEX'] for pid in patient_numbers])
    patient_zip3s = np.array([patient_lookup[pid]['PATIENT_ZIP3'] for pid in patient_numbers])
    patient_states = np.array([patient_lookup[pid]['PATIENT_STATE'] for pid in patient_numbers])

    diagnosis_codes_list = []
    for _ in tqdm(range(total_mx_events), desc="  Generating diagnosis codes"):
        num_dx = random.randint(DIAGNOSIS_CODES_MIN, DIAGNOSIS_CODES_MAX)
        diagnosis_codes_list.append(sample_diagnosis_codes(num_dx))

    mx_events_df = pd.DataFrame({
        'CLAIM_UUID': claim_uuids,
        'PATIENT_NUMBER': patient_numbers,
        'SERVICE_DATE': service_dates,
        'SERVICE_TO_DATE': service_to_dates,
        'PROCEDURE_CODE': procedure_codes_arr,
        'PROCEDURE_CODE_TYPE': procedure_code_types,
        'NDC11': ndc11_values,
        'UNITS': units,
        'UNIT_TYPE': unit_types,
        'RENDERING_NPI': rendering_npis,
        'REFERRING_NPI': referring_npis,
        'BILLING_NPI': billing_npis,
        'FACILITY_NPI': facility_npis,
        'VISIT_TYPE': visit_types,
        'INSURANCE_GROUP': insurance_groups,
        'DIAGNOSIS_CODES': diagnosis_codes_list,
        'ADMIT_DIAGNOSIS_CODE': admit_dx_values,
        'EVENT_SOURCE': event_sources,
        'PROVIDER_BILLED': provider_billed,
        'PLAN_ALLOWED': plan_allowed,
        'PLAN_PAID': plan_paid,
        'PATIENT_YOB': patient_yobs,
        'PATIENT_SEX': patient_sexes,
        'PATIENT_ZIP3': patient_zip3s,
        'PATIENT_STATE': patient_states,
    })
    print(f"  Generated MX events table: {len(mx_events_df):,} rows")

    # =========================================================================
    # Generate RX Events Table
    # =========================================================================
    print("\nGenerating RX events...")
    rx_patient_event_counts = np.random.randint(RX_EVENTS_MIN, RX_EVENTS_MAX + 1, size=NUM_PATIENTS)
    total_rx_events = rx_patient_event_counts.sum()
    print(f"  Total RX events: {total_rx_events:,}")

    rx_patient_numbers = np.repeat(patient_ids, rx_patient_event_counts)
    fill_dates = random_dates_array(DATE_START, DATE_END, total_rx_events)

    days_before = np.random.randint(0, 366, size=total_rx_events)
    rx_written_dates = [max(fd - timedelta(days=int(db)), DATE_START) for fd, db in zip(fill_dates, days_before)]

    pharmacy_npis = np.random.choice(PHARMACY_NPI_POOL, size=total_rx_events)
    prescriber_npis = np.random.choice(PRESCRIBER_NPI_POOL, size=total_rx_events)

    ndc_row_indices = np.random.randint(0, len(ndc_product_df), size=total_rx_events)
    ndc_rows = ndc_product_df.iloc[ndc_row_indices]
    ndc11_arr = ndc_rows['NDC11'].values
    brand_names = ndc_rows['BRAND_NAME'].values
    generic_names = ndc_rows['GENERIC_NAME'].values
    routes = ndc_rows['ROUTE'].values

    pharmacy_channels = np.random.choice(PHARMACY_CHANNELS, size=total_rx_events)
    transaction_results = np.random.choice(TRANSACTION_RESULTS, size=total_rx_events)
    transaction_numbers = np.random.randint(1, 101, size=total_rx_events)
    transaction_statuses = np.random.choice(TRANSACTION_STATUSES, size=total_rx_events)
    fill_numbers = np.random.randint(0, 100, size=total_rx_events)
    refills_authorized = np.random.randint(0, 100, size=total_rx_events)

    days_supply_mask = np.random.random(size=total_rx_events) < 0.95
    days_supply = np.where(
        days_supply_mask,
        np.random.randint(1, 91, size=total_rx_events),
        np.random.randint(91, 366, size=total_rx_events)
    )

    quantities = np.round(np.random.uniform(1.0, 500.0, size=total_rx_events), 2)
    diagnosis_code_arr = np.random.choice(icd10_codes, size=total_rx_events)

    pharmacy_submitted = np.round(np.random.uniform(10.0, 5000.0, size=total_rx_events), 2)
    rx_plan_paid = np.round(np.random.uniform(0.0, 1.0, size=total_rx_events) * pharmacy_submitted, 2)
    patient_responsibility = np.round(np.random.uniform(0.0, 500.0, size=total_rx_events), 2)
    patient_oop = np.round(patient_responsibility * np.random.uniform(0.8, 1.0, size=total_rx_events), 2)

    rx_claim_uuids = np.random.randint(100_000_000_000, 999_999_999_999, size=total_rx_events, dtype=np.int64)

    rx_insurance_groups = np.array([patient_lookup[pid]['INSURANCE_GROUP'] for pid in rx_patient_numbers])
    rx_patient_yobs = np.array([patient_lookup[pid]['PATIENT_YOB_DATE'] for pid in rx_patient_numbers])
    rx_patient_sexes = np.array([patient_lookup[pid]['PATIENT_SEX'] for pid in rx_patient_numbers])
    rx_patient_zip3s = np.array([patient_lookup[pid]['PATIENT_ZIP3'] for pid in rx_patient_numbers])
    rx_patient_states = np.array([patient_lookup[pid]['PATIENT_STATE'] for pid in rx_patient_numbers])

    rx_events_df = pd.DataFrame({
        'CLAIM_UUID': rx_claim_uuids,
        'PATIENT_NUMBER': rx_patient_numbers,
        'FILL_DATE': fill_dates,
        'PHARMACY_NPI': pharmacy_npis,
        'PHARMACY_CHANNEL': pharmacy_channels,
        'PRESCRIBER_NPI': prescriber_npis,
        'NDC11': ndc11_arr,
        'BRAND_NAME': brand_names,
        'GENERIC_NAME': generic_names,
        'ROUTE': routes,
        'DAYS_SUPPLY': days_supply,
        'QUANTITY': quantities,
        'DIAGNOSIS_CODE': diagnosis_code_arr,
        'TRANSACTION_RESULT': transaction_results,
        'TRANSACTION_NUMBER': transaction_numbers,
        'TRANSACTION_STATUS': transaction_statuses,
        'FILL_NUMBER': fill_numbers,
        'NUMBER_OF_REFILLS_AUTHORIZED': refills_authorized,
        'DATE_PRESCRIPTION_WRITTEN': rx_written_dates,
        'INSURANCE_GROUP': rx_insurance_groups,
        'PHARMACY_SUBMITTED_AMOUNT': pharmacy_submitted,
        'PLAN_PAID': rx_plan_paid,
        'PATIENT_RESPONSIBILITY': patient_responsibility,
        'PATIENT_OOP': patient_oop,
        'PATIENT_YOB': rx_patient_yobs,
        'PATIENT_SEX': rx_patient_sexes,
        'PATIENT_ZIP3': rx_patient_zip3s,
        'PATIENT_STATE': rx_patient_states,
    })
    print(f"  Generated RX events table: {len(rx_events_df):,} rows")

    # =========================================================================
    # Export to CSV
    # =========================================================================
    print("\nExporting to CSV...")
    demographics_df.to_csv(DATA_DIR / "demographics.csv", index=False)
    print("  Saved demographics.csv")

    geography_df.to_csv(DATA_DIR / "geography.csv", index=False)
    print("  Saved geography.csv")

    mortality_df.to_csv(DATA_DIR / "mortality.csv", index=False)
    print("  Saved mortality.csv")

    mx_events_df.to_csv(DATA_DIR / "mx_events.csv", index=False)
    print("  Saved mx_events.csv")

    rx_events_df.to_csv(DATA_DIR / "rx_events.csv", index=False)
    print("  Saved rx_events.csv")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"\nPatient Data (generated):")
    print(f"  demographics.csv:             {len(demographics_df):>10,} rows")
    print(f"  geography.csv:                {len(geography_df):>10,} rows")
    print(f"  mortality.csv:                {len(mortality_df):>10,} rows")
    print(f"  mx_events.csv:                {len(mx_events_df):>10,} rows")
    print(f"  rx_events.csv:                {len(rx_events_df):>10,} rows")
    print(f"\nCode Lookup Tables (pre-existing):")
    print(f"  icd10_codes.csv:              {len(icd10_df):>10,} rows")
    print(f"  procedure_codes.csv:          {len(procedures_df):>10,} rows")
    print(f"  ndc_products.csv:             {len(ndc_product_df):>10,} rows")
    print(f"\nOutput directory: {DATA_DIR}")


if __name__ == "__main__":
    main()
