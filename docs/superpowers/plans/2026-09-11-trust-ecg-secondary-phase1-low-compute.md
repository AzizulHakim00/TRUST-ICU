# TRUST-ECG Secondary Validation Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add low-compute secondary sensitivity, certification-uncertainty, Phase-1 estimability, overlap-audit, and reporting utilities around the frozen TRUST-ECG v0.4 primary experiment without modifying primary scientific behavior.

**Architecture:** Add new isolated `ecg_secondary_*` modules that consume verified aggregate primary artifacts and existing statistical helpers. The modules emit aggregate-only JSON/CSV summaries under a separate secondary-results namespace and never overwrite frozen Phase-0/Phase-1 artifacts. A small orchestration script and workflow validate provenance and generate publication-safe outputs.

**Tech Stack:** Python 3.11, NumPy, SciPy, scikit-learn, pandas only where already used, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-11-trust-ecg-secondary-validation-design.md`

## Global Constraints

- Frozen primary protocol, model architecture, label mapping, folds, external partition, primary seed, Phase-0 gate, and Phase-1 methods are immutable.
- Secondary outputs must carry protocol/report/model hashes and must never replace canonical primary artifacts.
- No record-level identifiers, raw waveforms, record-level probabilities, or unrestricted per-record attributions may be committed.
- Official primary seed remains `20260808`; additional-seed work is deferred to Phase 3.
- All sensitivity settings are explicitly secondary and must include the official gate unchanged as the reference condition.

---

### Task 1: Certification-envelope sensitivity and framework ablation

**Files:**
- Create: `tests/test_ecg_secondary_sensitivity.py`
- Create: `src/trust_icu/ecg_secondary_sensitivity.py`

**Interfaces:**
- Consumes: Phase-0 pair dictionaries containing `status`, support and `metrics`; existing certification metric semantics from `ecg_baseline.py`.
- Produces: `SecondaryEnvelope`, `classify_pair_secondary(...)`, `evaluate_envelope_grid(...)`, and `evaluate_framework_ablations(...)`.

- [ ] **Step 1: Write the failing tests**

```python
from trust_icu.ecg_secondary_sensitivity import (
    SecondaryEnvelope,
    classify_pair_secondary,
    evaluate_envelope_grid,
    evaluate_framework_ablations,
)


def _pair():
    return {
        "support": {"positive": 80, "negative": 120},
        "metrics": {
            "prevalence": 0.4,
            "pr_auc": 0.84,
            "pr_auc_to_prevalence_ratio": 2.1,
            "brier_skill_vs_prevalence": 0.08,
            "calibration_slope": 0.80,
            "calibration_intercept": 0.20,
        },
    }


def test_official_secondary_envelope_reproduces_certified_status():
    envelope = SecondaryEnvelope.official()
    assert classify_pair_secondary(_pair(), envelope) == "certified"


def test_grid_keeps_official_reference_condition():
    rows = evaluate_envelope_grid({"georgia": {"59118001": _pair()}})
    official = [row for row in rows if row["is_official"]]
    assert len(official) == 1
    assert official[0]["pr_ratio_min"] == 2.0
    assert official[0]["slope_deviation_max"] == 0.35
    assert official[0]["intercept_abs_max"] == 0.75
    assert official[0]["require_positive_brier_skill"] is True


def test_framework_ablation_does_not_mutate_input():
    matrix = {"georgia": {"59118001": _pair()}}
    before = repr(matrix)
    rows = evaluate_framework_ablations(matrix)
    assert rows
    assert repr(matrix) == before
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest tests/test_ecg_secondary_sensitivity.py -q`
Expected: import failure because `trust_icu.ecg_secondary_sensitivity` does not exist.

- [ ] **Step 3: Implement the minimal sensitivity module**

Implement an immutable `SecondaryEnvelope` dataclass with the official thresholds and a pure pair classifier that returns only the four existing statuses. Implement the fixed grid `{1.5,2.0,2.5,3.0} × {0.25,0.35,0.50} × {0.50,0.75,1.00} × {True,False}` plus the framework ablations defined in the approved spec. Preserve insufficient-support status before applying discrimination or calibration gates.

- [ ] **Step 4: Run focused tests and then existing ECG tests**

Run:
`pytest tests/test_ecg_secondary_sensitivity.py tests/test_ecg_baseline.py tests/test_ecg_phase0.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: add TRUST-ECG certification sensitivity analysis`

---

### Task 2: Certification uncertainty and Phase-1 estimability summaries

**Files:**
- Create: `tests/test_ecg_secondary_statistics.py`
- Create: `src/trust_icu/ecg_secondary_statistics.py`

**Interfaces:**
- Consumes: binary targets/probabilities for bootstrap uncertainty when available; existing Phase-1 aggregate repeat records/summaries.
- Produces: `bootstrap_gate_uncertainty(...)`, `wilson_rate_interval(...)`, and `summarize_phase1_estimability(...)`.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from trust_icu.ecg_secondary_statistics import (
    bootstrap_gate_uncertainty,
    summarize_phase1_estimability,
    wilson_rate_interval,
)


def test_gate_uncertainty_is_deterministic():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    p = np.array([0.05, 0.1, 0.2, 0.8, 0.7, 0.9, 0.75, 0.15])
    first = bootstrap_gate_uncertainty(y, p, repeats=50, seed=17)
    second = bootstrap_gate_uncertainty(y, p, repeats=50, seed=17)
    assert first == second
    assert 0 <= first["gate_satisfaction_rate"] <= 1


def test_wilson_rate_interval_bounds_rate():
    low, high = wilson_rate_interval(90, 100)
    assert low < 0.9 < high


def test_phase1_estimability_preserves_nonestimable_counts():
    summary = summarize_phase1_estimability(
        repeats_requested=100,
        estimable_repeats=83,
        recovered_repeats=75,
        nonestimable_reasons={"single_class_evaluation": 17},
    )
    assert summary["nonestimable_repeats"] == 17
    assert summary["recovery_rate_among_estimable"] == 75 / 83
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_ecg_secondary_statistics.py -q`
Expected: import failure for the new module.

- [ ] **Step 3: Implement minimal statistical helpers**

Reuse existing binary metric semantics, stratified bootstrap logic where both classes are required, and Wilson intervals. Return intervals for PR-ratio, calibration slope/intercept, Brier skill, and the proportion of valid bootstrap replicates that satisfy the official complete gate. Treat non-estimable replicates as reported exclusions, never as failures or successes.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_ecg_secondary_statistics.py tests/test_ecg_statistical_core.py tests/test_ecg_phase1.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: add TRUST-ECG certification uncertainty summaries`

---

### Task 3: Aggregate duplicate and near-duplicate screening primitives

**Files:**
- Create: `tests/test_ecg_secondary_overlap.py`
- Create: `src/trust_icu/ecg_secondary_overlap.py`

**Interfaces:**
- Consumes: in-memory standardized waveforms supplied by an execution adapter; partition labels supplied separately.
- Produces: `waveform_fingerprint(...)`, `cosine_similarity(...)`, and `summarize_cross_partition_overlap(...)` with aggregate-only output.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from trust_icu.ecg_secondary_overlap import (
    summarize_cross_partition_overlap,
    waveform_fingerprint,
)


def test_exact_duplicate_is_detected_without_identifier_output():
    x = np.arange(120, dtype=float).reshape(12, 10)
    summary = summarize_cross_partition_overlap([x], [x.copy()], similarity_threshold=0.9999)
    assert summary["exact_duplicate_pairs"] == 1
    assert summary["near_duplicate_pairs"] >= 1
    assert "record_id" not in repr(summary)


def test_fingerprint_is_deterministic_and_compact():
    x = np.ones((12, 5000), dtype=float)
    first = waveform_fingerprint(x)
    second = waveform_fingerprint(x)
    assert np.array_equal(first, second)
    assert first.ndim == 1
    assert first.size <= 512
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_ecg_secondary_overlap.py -q`
Expected: import failure for the new module.

- [ ] **Step 3: Implement minimal overlap logic**

Use a deterministic compact fingerprint based on per-lead downsampled/normalized summaries. Exact duplicate detection uses a content digest; near-duplicate screening uses cosine similarity on the compact fingerprint. Public summaries expose only counts, maximum similarity, threshold, and audit status.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_ecg_secondary_overlap.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: add TRUST-ECG cross-partition overlap audit`

---

### Task 4: Secondary aggregate report and privacy/provenance validation

**Files:**
- Create: `tests/test_ecg_secondary_reporting.py`
- Create: `src/trust_icu/ecg_secondary_reporting.py`
- Create: `scripts/run_open_ecg_secondary_phase1.py`

**Interfaces:**
- Consumes: verified Phase-0/Phase-1/statistical artifacts and outputs from Tasks 1-3.
- Produces: publication-safe CSV/JSON under a caller-supplied secondary output root plus `secondary_sha256_manifest.json`.

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path
from trust_icu.ecg_secondary_reporting import validate_public_secondary_payload


def test_privacy_validator_rejects_record_identifiers():
    try:
        validate_public_secondary_payload({"record_id": "A0001"})
    except ValueError as exc:
        assert "record" in str(exc).lower()
    else:
        raise AssertionError("record-level identifier was not rejected")


def test_privacy_validator_accepts_aggregate_counts():
    validate_public_secondary_payload({"certified": 9, "calibration_recovery_candidate": 9})
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_ecg_secondary_reporting.py -q`
Expected: import failure for the new module.

- [ ] **Step 3: Implement report/provenance layer**

Validate exact primary protocol/report/model hashes before assembling outputs. Recursively reject identifier-like keys and raw waveform/probability arrays from public payloads. Write deterministic JSON/CSV and a SHA-256 manifest. The runner supports `--dry-run` without primary artifacts and prints the fixed analysis configuration.

- [ ] **Step 4: Verify**

Run:
`pytest tests/test_ecg_secondary_reporting.py tests/test_ecg_secondary_sensitivity.py tests/test_ecg_secondary_statistics.py tests/test_ecg_secondary_overlap.py -q`
`python scripts/run_open_ecg_secondary_phase1.py --dry-run`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: assemble TRUST-ECG low-compute secondary evidence`

---

### Task 5: CI/workflow integration for Phase 1

**Files:**
- Create: `.github/workflows/open-ecg-secondary-validation-phase1.yml`
- Modify: `.github/workflows/ci.yml` only if necessary to add the new runner dry-run; do not alter frozen ECG scientific workflows.

**Interfaces:**
- Consumes: repository source and, for real execution, verified primary aggregate artifacts/caches.
- Produces: test evidence and an aggregate-only secondary artifact.

- [ ] **Step 1: Add workflow-level validation assertions before real execution**

The workflow must run the focused unit tests and `--dry-run` first, verify frozen protocol hashes, and use a separate `secondary_results/trust_ecg_v04` output root.

- [ ] **Step 2: Ensure no primary artifact path is writable by the secondary runner**

Use distinct work/output directories and fail if the configured secondary output path resolves inside a canonical primary results directory.

- [ ] **Step 3: Run PR CI**

Expected: Ruff, pytest, protocol validation, and synthetic end-to-end CI all pass.

- [ ] **Step 4: Run secondary Phase-1 workflow in dry-run/unit mode**

Expected: workflow succeeds without waveform download or model retraining.

- [ ] **Step 5: Commit**

Commit message: `ci: add TRUST-ECG secondary validation phase 1`

---

## Plan self-review

- Spec coverage: Phase 1 intentionally covers envelope sensitivity, framework ablation, certification uncertainty, Phase-1 estimability, overlap audit, aggregate reporting, provenance, and privacy. XAI/robustness and multi-seed training are deferred to dedicated Phase 2 and Phase 3 plans.
- Placeholder scan: no TBD/TODO placeholders.
- Type consistency: new modules communicate using plain dictionaries/NumPy arrays and named pure functions; no task depends on an undefined production interface.
- Scientific safety: no task changes `schemas/open_ecg_protocol.yaml`, `ecg_resnet.py`, frozen Phase-0/Phase-1 scientific behavior, labels, calibration method, or primary gate.
