"""Aggregate label-concordance audit for original PTB-XL development labels.

TRUST-ECG develops models directly on the original PTB-XL v1.0.1 release so that official
patient identifiers and stratified folds are available without attempting to reverse-map the
Challenge-renamed PTB-XL records. This module checks that the prespecified SCP-ECG statements
produce the same aggregate positive counts as the corresponding SNOMED-CT labels observed in
the public Challenge 2020 PTB-XL headers before the development representation is unlocked.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# These seven mappings were fixed before model-performance inspection. The Challenge counts are
# aggregate label evidence from the real header-only audit, not model results.
PTBXL_SCP_TO_CHALLENGE = {
    "59118001": {"abbreviation": "RBBB", "scp_code": "CRBBB", "challenge_positive_count": 542},
    "164889003": {"abbreviation": "AF", "scp_code": "AFIB", "challenge_positive_count": 1514},
    "164909002": {"abbreviation": "LBBB", "scp_code": "CLBBB", "challenge_positive_count": 536},
    "270492004": {"abbreviation": "IAVB", "scp_code": "1AVB", "challenge_positive_count": 797},
    "284470004": {"abbreviation": "PAC", "scp_code": "PAC", "challenge_positive_count": 555},
    "426783006": {"abbreviation": "NSR", "scp_code": "SR", "challenge_positive_count": 18092},
    "427084000": {"abbreviation": "STach", "scp_code": "STACH", "challenge_positive_count": 826},
}


@dataclass(frozen=True)
class LabelConcordanceRow:
    canonical_code: str
    abbreviation: str
    scp_code: str
    challenge_positive_count: int
    original_ptbxl_key_present_count: int
    original_ptbxl_positive_likelihood_count: int
    exact_key_present_match: bool
    exact_positive_likelihood_match: bool


@dataclass(frozen=True)
class PtbxlLabelConcordanceAudit:
    audit_version: str
    ptbxl_rows: int
    unique_ecg_ids: int
    unique_patients: int
    folds_present: tuple[int, ...]
    patients_spanning_multiple_folds: int
    label_rows: tuple[LabelConcordanceRow, ...]
    selected_count_semantics: str
    all_labels_exactly_concordant: bool
    ready_for_original_ptbxl_development: bool
    blockers: tuple[str, ...]
    audit_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_hash(payload: dict[str, Any], key: str) -> str:
    material = dict(payload)
    material[key] = ""
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _parse_scp_codes(raw: str) -> dict[str, float]:
    parsed = ast.literal_eval(raw)
    if not isinstance(parsed, dict):
        raise ValueError("PTB-XL scp_codes must parse to a dictionary.")
    result: dict[str, float] = {}
    for key, value in parsed.items():
        code = str(key)
        likelihood = float(value)
        if not (likelihood == likelihood):
            raise ValueError("PTB-XL scp_codes contains NaN likelihood.")
        result[code] = likelihood
    return result


def build_ptbxl_label_concordance_audit(metadata_csv: str | Path) -> PtbxlLabelConcordanceAudit:
    path = Path(metadata_csv).expanduser().resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"ecg_id", "patient_id", "strat_fold", "scp_codes"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("PTB-XL metadata lacks ecg_id, patient_id, strat_fold, or scp_codes.")
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError("PTB-XL metadata is empty.")
    ecg_ids = [int(row["ecg_id"]) for row in rows]
    if len(set(ecg_ids)) != len(ecg_ids):
        raise ValueError("PTB-XL metadata contains duplicate ecg_id values.")

    patient_folds: dict[str, set[int]] = {}
    key_counts = {code: 0 for code in PTBXL_SCP_TO_CHALLENGE}
    positive_counts = {code: 0 for code in PTBXL_SCP_TO_CHALLENGE}
    folds: set[int] = set()

    for row in rows:
        patient = str(row["patient_id"])
        fold = int(row["strat_fold"])
        folds.add(fold)
        patient_folds.setdefault(patient, set()).add(fold)
        scp_codes = _parse_scp_codes(str(row["scp_codes"]))
        for canonical, spec in PTBXL_SCP_TO_CHALLENGE.items():
            scp_code = str(spec["scp_code"])
            if scp_code in scp_codes:
                key_counts[canonical] += 1
                if scp_codes[scp_code] > 0:
                    positive_counts[canonical] += 1

    leaking_patients = sum(1 for patient_folds_set in patient_folds.values() if len(patient_folds_set) > 1)
    label_rows: list[LabelConcordanceRow] = []
    for canonical, spec in PTBXL_SCP_TO_CHALLENGE.items():
        target = int(spec["challenge_positive_count"])
        label_rows.append(
            LabelConcordanceRow(
                canonical_code=canonical,
                abbreviation=str(spec["abbreviation"]),
                scp_code=str(spec["scp_code"]),
                challenge_positive_count=target,
                original_ptbxl_key_present_count=key_counts[canonical],
                original_ptbxl_positive_likelihood_count=positive_counts[canonical],
                exact_key_present_match=key_counts[canonical] == target,
                exact_positive_likelihood_match=positive_counts[canonical] == target,
            )
        )

    all_key_exact = all(row.exact_key_present_match for row in label_rows)
    all_positive_exact = all(row.exact_positive_likelihood_match for row in label_rows)
    if all_key_exact:
        semantics = "scp_key_present"
        concordant = True
    elif all_positive_exact:
        semantics = "scp_likelihood_gt_zero"
        concordant = True
    else:
        semantics = "unresolved"
        concordant = False

    blockers: list[str] = []
    if len(rows) != 21837:
        blockers.append("ptbxl_v1_0_1_row_count_mismatch")
    if sorted(folds) != list(range(1, 11)):
        blockers.append("ptbxl_expected_folds_missing")
    if leaking_patients:
        blockers.append("ptbxl_patient_fold_leakage_detected")
    if not concordant:
        blockers.append("ptbxl_scp_to_challenge_label_counts_not_exactly_concordant")

    payload: dict[str, Any] = {
        "audit_version": "0.1.0",
        "ptbxl_rows": len(rows),
        "unique_ecg_ids": len(set(ecg_ids)),
        "unique_patients": len(patient_folds),
        "folds_present": sorted(folds),
        "patients_spanning_multiple_folds": leaking_patients,
        "label_rows": [asdict(row) for row in label_rows],
        "selected_count_semantics": semantics,
        "all_labels_exactly_concordant": concordant,
        "ready_for_original_ptbxl_development": not blockers,
        "blockers": blockers,
        "audit_sha256": "",
    }
    payload["audit_sha256"] = _canonical_hash(payload, "audit_sha256")
    return PtbxlLabelConcordanceAudit(
        audit_version=str(payload["audit_version"]),
        ptbxl_rows=int(payload["ptbxl_rows"]),
        unique_ecg_ids=int(payload["unique_ecg_ids"]),
        unique_patients=int(payload["unique_patients"]),
        folds_present=tuple(int(value) for value in payload["folds_present"]),
        patients_spanning_multiple_folds=int(payload["patients_spanning_multiple_folds"]),
        label_rows=tuple(label_rows),
        selected_count_semantics=str(payload["selected_count_semantics"]),
        all_labels_exactly_concordant=bool(payload["all_labels_exactly_concordant"]),
        ready_for_original_ptbxl_development=bool(payload["ready_for_original_ptbxl_development"]),
        blockers=tuple(str(value) for value in payload["blockers"]),
        audit_sha256=str(payload["audit_sha256"]),
    )


def write_ptbxl_label_concordance_audit(audit: PtbxlLabelConcordanceAudit, output: str | Path) -> None:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
