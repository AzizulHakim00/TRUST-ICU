# Credentialed Phase 0 Execution

This workflow runs only inside an authorized environment with local access to MIMIC-IV v3.1
or eICU-CRD v2.0. The repository never reads a committed credential file and never prints
patient-level rows.

## Install

```bash
git checkout phase-0-feasibility
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,db]"
```

## Configure secrets outside Git

Use a PostgreSQL role with read access to the source schemas and create/drop access only to the
local `trust_icu_work` and `trust_icu_local` schemas.

```bash
export TRUST_ICU_POSTGRES_DSN='postgresql://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require'
export TRUST_ICU_OUTPUT_ROOT='/secure/trust_icu_outputs'
```

Prefer a password file, secret manager or protected shell session where available. Do not paste
the DSN into notebooks, issues, logs or committed files.

## Credential-free preflight

```bash
python scripts/run_credentialed_extract.py \
  --dataset mimic_iv_3_1 \
  --dry-run
```

The dry run prints only adapter paths, statuses and SQL SHA-256 values.

## MIMIC-IV execution

The MIMIC adapter is implemented but still requires local credentialed validation.

```bash
python scripts/run_credentialed_extract.py \
  --dataset mimic_iv_3_1
```

The runner:

1. validates the public adapter manifest and contracts;
2. materializes `trust_icu_work.mimic_stays`;
3. materializes canonical events and observations;
4. runs aggregate database-side validation;
5. writes `canonical_audit.json`;
6. exports compressed CSV files only after the critical audit passes;
7. writes a hashed `credentialed_run_report.json`.

Output:

```text
$TRUST_ICU_OUTPUT_ROOT/mimic_iv_3_1/
  canonical_audit.json
  credentialed_run_report.json
  mimic_iv_3_1_stays.csv.gz
  mimic_iv_3_1_events.csv.gz
  mimic_iv_3_1_observations.csv.gz
```

Existing non-empty output directories are protected. Use `--overwrite` only after deliberately
archiving or deleting the previous run.

## eICU preparation and execution

First create empty review tables:

```bash
python scripts/run_credentialed_extract.py \
  --dataset eicu_crd_2_0 \
  --prepare-eicu-mappings
```

Run the vocabulary-discovery SQL, review frequencies and units, obtain clinical approval, and
load only approved rows as `status='locked'`. The runner requires at least one locked positive
mapping for each outcome and at least one locked local feature mapping.

After review:

```bash
python scripts/run_credentialed_extract.py \
  --dataset eicu_crd_2_0 \
  --allow-reviewed-eicu
```

The explicit flag does not approve mappings. It only confirms that the user intends to run the
database checks; missing locked mappings still stop execution.

## Failure behavior

The runner fails closed when it detects:

- incorrect dataset identifiers;
- duplicate stays or observations;
- invalid ICU or event intervals;
- unknown tasks or variables;
- unit mismatches;
- unlinked rows;
- predictors before ICU admission or at/after the six-hour landmark;
- missing source provenance;
- missing or non-finite values;
- empty observation extracts;
- incomplete eICU reviewed mappings.

If the aggregate audit fails, patient-level exports are not written. The aggregate audit JSON is
retained locally for debugging.

## Security and governance

- Never commit the output directory.
- Never upload canonical extracts to GitHub, chat systems or public cloud storage not covered by
  the applicable data-use agreement.
- Review aggregate export permissions before moving audit reports outside the secure environment.
- Rotate credentials immediately if a DSN is exposed.
- The execution report contains database version, SQL hashes, row counts and file hashes, but no
  DSN or patient rows.
