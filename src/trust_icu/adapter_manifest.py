"""Validation for the public source-adapter execution manifest."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

_ALLOWED_DATASETS = {"mimic_iv_3_1", "eicu_crd_2_0"}
_ALLOWED_STATUSES = {
    "implemented_pending_credentialed_execution",
    "templates_require_reviewed_local_mappings",
    "locally_validated",
    "locked",
}
_REQUIRED_EXPORTS = {"stays", "events", "observations"}
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class AdapterDatasetReport:
    dataset: str
    status: str
    files_present: bool
    local_mappings_required: bool
    credentialed_validation_required: bool
    ready_for_credentialed_execution: bool
    missing_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdapterManifestReport:
    version: str
    datasets: tuple[AdapterDatasetReport, ...]
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "valid": self.valid,
            "datasets": [dataset.to_dict() for dataset in self.datasets],
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Adapter manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Adapter-manifest root must be a mapping.")
    return raw


def _safe_relative_path(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a repository-relative safe path: {value!r}.")
    return path


def validate_adapter_manifest(raw: dict[str, Any], repo_root: str | Path) -> AdapterManifestReport:
    """Validate manifest structure and verify all declared public files exist."""

    root = Path(repo_root)
    version = str(raw.get("manifest_version", ""))
    if not version:
        raise ValueError("manifest_version is required.")

    for contract_key in ("canonical_contract", "feature_contract", "outcome_contract"):
        value = raw.get(contract_key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{contract_key} must be a repository-relative path.")
        contract_path = _safe_relative_path(value, contract_key)
        if not (root / contract_path).is_file():
            raise FileNotFoundError(f"Declared contract does not exist: {contract_path}")

    datasets = raw.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("datasets must be a mapping.")
    if set(datasets) != _ALLOWED_DATASETS:
        raise ValueError(f"datasets must equal {sorted(_ALLOWED_DATASETS)}.")

    reports: list[AdapterDatasetReport] = []
    for dataset, spec in datasets.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Dataset specification must be a mapping: {dataset}")
        status = str(spec.get("status", ""))
        if status not in _ALLOWED_STATUSES:
            raise ValueError(f"Unsupported adapter status for {dataset}: {status!r}.")
        if spec.get("dialect") != "postgresql":
            raise ValueError(f"Phase 0 adapter dialect must be postgresql for {dataset}.")
        commit = str(spec.get("upstream_commit", ""))
        if not _HEX40.fullmatch(commit):
            raise ValueError(f"upstream_commit must be a 40-character lowercase SHA: {dataset}.")
        if spec.get("output_dataset_id") != dataset:
            raise ValueError(f"output_dataset_id must equal the dataset key for {dataset}.")

        execution = spec.get("execution_order")
        if not isinstance(execution, dict) or set(execution) != _REQUIRED_EXPORTS:
            raise ValueError(
                f"{dataset}.execution_order must define exactly {sorted(_REQUIRED_EXPORTS)}."
            )
        declared_paths: list[Path] = []
        for export_name in ("stays", "events", "observations"):
            value = execution[export_name]
            if not isinstance(value, str) or not value:
                raise ValueError(f"Missing {dataset}.{export_name} adapter path.")
            declared_paths.append(_safe_relative_path(value, f"{dataset}.{export_name}"))

        requires_maps = bool(spec.get("requires_local_vocabulary_tables"))
        templates = spec.get("local_mapping_templates", {})
        if requires_maps:
            if not isinstance(templates, dict) or set(templates) != {"features", "outcomes"}:
                raise ValueError(
                    f"{dataset} requires features and outcomes local mapping templates."
                )
            for name, value in templates.items():
                if not isinstance(value, str) or not value:
                    raise ValueError(f"Invalid {dataset}.{name} mapping template path.")
                declared_paths.append(_safe_relative_path(value, f"{dataset}.{name}"))
        elif templates:
            raise ValueError(f"{dataset} declares mapping templates but does not require them.")

        missing = tuple(str(path) for path in declared_paths if not (root / path).is_file())
        files_present = not missing
        ready = files_present and status in {
            "implemented_pending_credentialed_execution",
            "locally_validated",
            "locked",
        }
        reports.append(
            AdapterDatasetReport(
                dataset=dataset,
                status=status,
                files_present=files_present,
                local_mappings_required=requires_maps,
                credentialed_validation_required=bool(
                    spec.get("credentialed_validation_required")
                ),
                ready_for_credentialed_execution=ready,
                missing_paths=missing,
            )
        )

    return AdapterManifestReport(
        version=version,
        datasets=tuple(reports),
        valid=all(report.files_present for report in reports),
    )


def load_and_validate_adapter_manifest(
    path: str | Path,
    repo_root: str | Path | None = None,
) -> AdapterManifestReport:
    """Load and validate the adapter manifest relative to the repository root."""

    manifest_path = Path(path)
    root = Path(repo_root) if repo_root is not None else manifest_path.resolve().parents[1]
    return validate_adapter_manifest(_load_yaml(manifest_path), root)
