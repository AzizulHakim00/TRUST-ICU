# Outcome Evidence and Private Runtime Locking

Phase 0 model training remains blocked until the exact installed MIMIC-IV and eICU extracts have been locally validated and a clinical reviewer confirms that both databases represent the same incident organ-support estimand.

## 1. Build aggregate summaries

After each canonical adapter has passed its credentialed audit:

```bash
python scripts/build_outcome_validation_summary.py \
  --dataset mimic_iv_3_1 \
  --run-dir /secure/trust_icu_outputs/mimic_iv_3_1 \
  --output /secure/outcome_review/mimic_outcome_summary.json

python scripts/build_outcome_validation_summary.py \
  --dataset eicu_crd_2_0 \
  --run-dir /secure/trust_icu_outputs/eicu_crd_2_0 \
  --output /secure/outcome_review/eicu_outcome_summary.json
```

Each summary is aggregate-only and contains:

- canonical stay count;
- event rows and event stays;
- support active before the six-hour landmark;
- incident events in `[6 h, 18 h)` before ICU discharge;
- incident hospital coverage;
- source-table and source-code counts;
- invalid interval count;
- hashes of the credentialed run and audit.

The summary does not contain candidate terms, patient identifiers, hospital identifiers or row-level examples.

## 2. Review eICU vocabulary inside the credentialed environment

Run `sql/eicu/00_outcome_vocabulary_discovery.sql`, classify every candidate term and populate `trust_icu_local.eicu_outcome_map`. Only clinically reviewed mappings with `status='locked'` may participate in the canonical eICU event export.

For each task, record:

- number of candidate terms reviewed;
- number of unresolved candidate terms;
- number of locked positive mappings;
- rationale for exclusions and corroborating-only terms;
- hospital-interface coverage limitations.

## 3. Complete reviewer evidence

Copy `templates/outcome_validation_evidence.template.json` to a secure directory outside the repository. Replace all placeholders and provide both:

- a `data_reviewer` who verified extraction, frequencies and provenance;
- a `clinical_reviewer` who approved the clinical interpretation and cross-database equivalence.

Approval is task-specific. Each task must confirm:

- the same clinical event;
- equivalent exclusion of support active in `[0 h, 6 h)`;
- identical `[6 h, 18 h)` boundaries;
- the same primary prediction estimand;
- zero unresolved eICU candidate terms;
- at least one locked positive mapping;
- nonzero incident events in both databases;
- no invalid event intervals.

## 4. Create a private runtime context

```bash
python scripts/finalize_outcome_contracts.py \
  --evidence /secure/outcome_review/outcome_validation_evidence.json \
  --mimic-summary /secure/outcome_review/mimic_outcome_summary.json \
  --eicu-summary /secure/outcome_review/eicu_outcome_summary.json \
  --output-root /secure/trust_icu_locked_runtime
```

The command returns exit code `2` and writes a blocker report when evidence is incomplete. It creates a locked runtime contract only when every gate passes.

Successful output:

```text
/secure/trust_icu_locked_runtime/
  outcome_lock_report.json
  configs/feasibility.yaml
  schemas/phase0_features.yaml
  schemas/outcome_contracts.yaml
```

The public `schemas/outcome_contracts.yaml` is never modified. The private runtime contract contains evidence hashes and reviewer metadata.

## 5. Run Phase 0 with the private runtime root

```bash
python scripts/run_phase0_baselines.py \
  --repo-root /secure/trust_icu_locked_runtime \
  --mimic-run-dir /secure/trust_icu_outputs/mimic_iv_3_1 \
  --eicu-run-dir /secure/trust_icu_outputs/eicu_crd_2_0 \
  --output-root /secure/trust_icu_phase0_results
```

Do not commit the evidence, summaries, locked runtime context or resulting Phase 0 artifacts. A reviewer must not approve evidence solely to bypass the training guard.
