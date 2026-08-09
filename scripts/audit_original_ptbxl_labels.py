#!/usr/bin/env python3
"""Audit original PTB-XL SCP labels against Challenge PTB-XL aggregate label counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trust_icu.ecg_ptbxl_labels import (
    PTBXL_SCP_TO_CHALLENGE,
    build_ptbxl_label_concordance_audit,
    write_ptbxl_label_concordance_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl-metadata", help="PTB-XL v1.0.1 ptbxl_database.csv")
    parser.add_argument("--output", default="ptbxl_label_concordance_audit.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "original_ptbxl_scp_to_challenge_label_concordance",
                    "candidate_mapping": PTBXL_SCP_TO_CHALLENGE,
                    "gate": "all_seven_labels_must_match_exactly_under_one_count_semantics",
                    "model_performance_touched": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.ptbxl_metadata:
        raise SystemExit("--ptbxl-metadata is required outside --dry-run")
    audit = build_ptbxl_label_concordance_audit(args.ptbxl_metadata)
    write_ptbxl_label_concordance_audit(audit, args.output)
    print(
        json.dumps(
            {
                "ready_for_original_ptbxl_development": audit.ready_for_original_ptbxl_development,
                "selected_count_semantics": audit.selected_count_semantics,
                "all_labels_exactly_concordant": audit.all_labels_exactly_concordant,
                "label_rows": [
                    {
                        "canonical_code": row.canonical_code,
                        "abbreviation": row.abbreviation,
                        "scp_code": row.scp_code,
                        "challenge_positive_count": row.challenge_positive_count,
                        "original_ptbxl_key_present_count": row.original_ptbxl_key_present_count,
                        "original_ptbxl_positive_likelihood_count": row.original_ptbxl_positive_likelihood_count,
                        "exact_key_present_match": row.exact_key_present_match,
                        "exact_positive_likelihood_match": row.exact_positive_likelihood_match,
                    }
                    for row in audit.label_rows
                ],
                "blockers": audit.blockers,
                "audit_sha256": audit.audit_sha256,
                "output": str(Path(args.output).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if args.require_ready and not audit.ready_for_original_ptbxl_development else 0


if __name__ == "__main__":
    raise SystemExit(main())
