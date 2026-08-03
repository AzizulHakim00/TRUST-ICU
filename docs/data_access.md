# Data Access and Governance Checklist

TRUST-ICU uses credentialed clinical databases. This public repository contains code and aggregate documentation only.

## MIMIC-IV v3.1

Role: development and temporal validation.

Required before extraction:

- PhysioNet credentialed-user status;
- CITI Data or Specimens Only Research training;
- signed PhysioNet Credentialed Health Data Use Agreement;
- approved MIMIC-IV access;
- local or BigQuery access to the v3.1 schemas;
- confirmation that the user understands restrictions on derived data and trained models.

Expected schemas when using BigQuery:

- `mimiciv_v3_1_hosp`
- `mimiciv_v3_1_icu`

## eICU-CRD v2.0

Role: primary fully external, multi-hospital validation.

Required before extraction:

- PhysioNet credentialed-user status;
- CITI Data or Specimens Only Research training;
- signed project data-use agreement;
- approved eICU-CRD access.

An open eICU demo exists and may be used only to test schema handling and unit tests. Demo metrics are never scientific results.

## AmsterdamUMCdb v1.5.0

Role: later European external validation.

Access is requested through the Amsterdam Medical Data Science access process. Version 1.5.0 is represented in OMOP CDM 5.4. Access approval and the applicable agreement must be completed before any patient-level processing.

## HiRID v1.1.1

Role: later high-time-resolution European validation.

Required before extraction:

- PhysioNet credentialed-user status;
- CITI training;
- signed data-use agreement;
- author-approved project request with a specific research question.

## Local environment variables

Use environment variables rather than hard-coded paths:

```bash
export TRUST_ICU_MIMIC_ROOT=/secure/path/to/mimiciv/3.1
export TRUST_ICU_EICU_ROOT=/secure/path/to/eicu/2.0
export TRUST_ICU_AMSTERDAM_ROOT=/secure/path/to/amsterdamumcdb/1.5.0
export TRUST_ICU_HIRID_ROOT=/secure/path/to/hirid/1.1.1
export TRUST_ICU_OUTPUT_ROOT=/secure/path/to/trust_icu_outputs
```

For BigQuery, use the platform's standard authenticated client configuration. Never commit service-account keys.

## Prohibited repository content

Do not commit or attach:

- raw database tables;
- patient-level cohort extracts;
- row-level predictions;
- timestamps or identifiers derived from restricted data;
- credentials, cookies or access tokens;
- model checkpoints trained on credentialed data unless release is explicitly allowed;
- screenshots exposing patient-level records.

## Allowed public artifacts

Subject to the relevant agreements:

- SQL and Python extraction code;
- synthetic fixtures;
- code lists and harmonization logic without patient records;
- aggregate cohort counts;
- aggregate performance tables and figures;
- configuration and provenance manifests;
- reporting checklists.

## Access-status record

Before Phase 0 modelling, create a local `access_status.yaml` from the following template. It is ignored by Git and must not include credentials.

```yaml
mimic_iv_3_1:
  approved: false
  storage_mode: null
  schema_verified: false

eicu_crd_2_0:
  approved: false
  storage_mode: null
  schema_verified: false

amsterdamumcdb_1_5_0:
  approved: false
  schema_verified: false

hirid_1_1_1:
  approved: false
  schema_verified: false
```

Phase 0 model training must not start until both MIMIC-IV and eICU access and schema verification are complete.
