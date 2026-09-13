from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_open_ecg_secondary_seed_waveform.py"
SPEC = importlib.util.spec_from_file_location("secondary_seed_waveform", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _state(seed: int) -> dict[str, object]:
    expected = MODULE.SEED_EXPECTATIONS[seed]
    return {
        "report": {
            "model_sha256": expected["model_sha256"],
            "report_sha256": expected["report_sha256"],
            "protocol_sha256": expected["protocol_sha256"],
            "model_index_sha256": "index",
            "label_manifest_sha256": "labels",
        },
        "stats": SimpleNamespace(stats_sha256="norm"),
    }


def test_seed09_provenance_is_secondary_and_does_not_replace_primary() -> None:
    state = _state(20260809)
    MODULE._verify_analysis_seed(state, 20260809)
    provenance = MODULE._analysis_provenance(state, 20260809)

    assert provenance["analysis_seed"] == 20260809
    assert provenance["analysis_model_role"] == "predeclared_secondary_resnet_sensitivity_model"
    assert provenance["model_sha256"] == MODULE.SEED_EXPECTATIONS[20260809]["model_sha256"]
    assert MODULE.HISTORICAL_PRIMARY["seed"] == 20260808
    assert MODULE.HISTORICAL_PRIMARY["model_sha256"].startswith("1dc5a0eb")


def test_seed10_provenance_is_locked_to_authoritative_artifact() -> None:
    state = _state(20260810)
    MODULE._verify_analysis_seed(state, 20260810)
    provenance = MODULE._analysis_provenance(state, 20260810)

    assert provenance["analysis_seed"] == 20260810
    assert provenance["model_sha256"] == "0cb57b6bcc7c124666d10def9c83cadc0260846392d81102174d14e6d426ce95"
    assert provenance["phase0_report_sha256"] == "31a281e8b77504038826acb5ad4be6b2e000aac7335582041f20120606836493"


def test_seed_identity_mismatch_fails_closed() -> None:
    state = _state(20260809)
    state["report"]["model_sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        MODULE._verify_analysis_seed(state, 20260809)
