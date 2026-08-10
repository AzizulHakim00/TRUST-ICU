from __future__ import annotations

import hashlib
from pathlib import Path

from trust_icu import ecg_index


def _audit_style_hash(source_root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        paths,
        key=lambda item: (ecg_index._numeric_record_id(item.stem), str(item)),
    ):
        relative = path.relative_to(source_root).as_posix()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _legacy_lexicographic_hash(source_root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(source_root).as_posix()):
        relative = path.relative_to(source_root).as_posix()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def test_nested_challenge_groups_match_waveform_audit_record_order(tmp_path: Path) -> None:
    source_root = tmp_path / "georgia"
    paths = [
        source_root / "g1" / "E00001.hea",
        source_root / "g2" / "E00002.hea",
        source_root / "g10" / "E00010.hea",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((path.stem + "-header").encode("utf-8"))

    expected = _audit_style_hash(source_root, paths)
    observed = ecg_index._hash_files(source_root, list(reversed(paths)))

    assert observed == expected
    assert _legacy_lexicographic_hash(source_root, paths) != expected


def test_canonical_corpus_hash_still_detects_byte_mutation(tmp_path: Path) -> None:
    source_root = tmp_path / "cpsc_2018"
    paths = [
        source_root / "g1" / "A0001.mat",
        source_root / "g2" / "A0002.mat",
        source_root / "g10" / "A0010.mat",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((path.stem + "-waveform").encode("utf-8"))

    before = ecg_index._hash_files(source_root, paths)
    paths[1].write_bytes(paths[1].read_bytes() + b"tamper")
    after = ecg_index._hash_files(source_root, paths)

    assert after != before
