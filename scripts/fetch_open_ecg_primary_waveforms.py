#!/usr/bin/env python3
"""Resumably fetch TRUST-ECG v0.4 primary waveform corpora from PhysioNet.

Downloads only original PTB-XL v1.0.1 records500 plus the three Challenge 2020 external domains.
Challenge-renamed PTB-XL waveforms, PTB, INCART, and the hidden test set are never downloaded by
this tool. Final files are atomically renamed only after a complete HTTP response; interrupted
transfers remain as ``.partial`` files and are resumed with HTTP Range requests when supported.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
import re
import threading
import time
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path

PTB_BASE = "https://physionet.org/files/ptb-xl/1.0.1/records500/"
CHALLENGE_BASE = "https://physionet.org/files/challenge-2020/1.0.2/training"
PTB_METADATA_URL = "https://physionet.org/files/ptb-xl/1.0.1/ptbxl_database.csv"
EXTERNAL_COUNTS = {"georgia": 10344, "cpsc_2018": 6877, "cpsc_2018_extra": 3453}
PTB_COUNT = 21837
_USER_AGENT = "TRUST-ECG-waveform-fetch/0.4 (+https://github.com/AzizulHakim00/TRUST-ICU)"
_GROUP_RE = re.compile(r"^g\d+/$")
_PTB_DIR_RE = re.compile(r"^\d{5}/$")
_THREAD_LOCAL = threading.local()
_CHUNK = 1024 * 1024


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def _validate_url(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "physionet.org" or parsed.username or parsed.password:
        raise ValueError(f"Only https://physionet.org URLs are permitted: {url!r}")
    return parsed


def _connection(timeout: float) -> http.client.HTTPSConnection:
    connection = getattr(_THREAD_LOCAL, "connection", None)
    if connection is None:
        connection = http.client.HTTPSConnection("physionet.org", timeout=timeout)
        _THREAD_LOCAL.connection = connection
    else:
        connection.timeout = timeout
    return connection


def _drop_connection() -> None:
    connection = getattr(_THREAD_LOCAL, "connection", None)
    if connection is not None:
        try:
            connection.close()
        finally:
            _THREAD_LOCAL.connection = None


def _request(url: str, *, headers: dict[str, str] | None = None, timeout: float = 90.0):
    parsed = _validate_url(url)
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    connection = _connection(timeout)
    request_headers = {"User-Agent": _USER_AGENT, "Accept": "*/*"}
    if headers:
        request_headers.update(headers)
    connection.request("GET", target, headers=request_headers)
    return connection.getresponse()


def _read_small(url: str, *, attempts: int = 5) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = _request(url, timeout=60.0)
            data = response.read()
            if response.status != 200 or not data:
                raise RuntimeError(f"HTTP {response.status} or empty response for {url}")
            return data
        except (OSError, TimeoutError, http.client.HTTPException, RuntimeError) as exc:
            last_error = exc
            _drop_connection()
            if attempt + 1 < attempts:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"Failed to fetch {url}") from last_error


def _hrefs(url: str) -> list[str]:
    parser = _HrefParser()
    parser.feed(_read_small(url).decode("utf-8", errors="replace"))
    return parser.hrefs


def _subdirs(url: str, pattern: re.Pattern[str]) -> list[str]:
    result = []
    for href in _hrefs(url):
        parsed = urllib.parse.urlparse(href)
        if not parsed.scheme and not parsed.netloc and pattern.fullmatch(parsed.path):
            result.append(parsed.path)
    values = sorted(set(result))
    if not values:
        raise RuntimeError(f"No expected subdirectories found at {url}")
    return values


def _files(url: str, suffixes: tuple[str, ...]) -> list[str]:
    names = []
    for href in _hrefs(url):
        parsed = urllib.parse.urlparse(href)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            continue
        name = Path(parsed.path).name
        if name and name not in {".", ".."} and name.endswith(suffixes):
            names.append(name)
    return sorted(set(names))


def _ptb_jobs(root: Path) -> list[tuple[str, Path]]:
    jobs: list[tuple[str, Path]] = []
    for directory in _subdirs(PTB_BASE, _PTB_DIR_RE):
        url = urllib.parse.urljoin(PTB_BASE, directory)
        names = _files(url, (".hea", ".dat"))
        stems = {Path(name).stem for name in names if name.endswith(".hea")}
        dat_stems = {Path(name).stem for name in names if name.endswith(".dat")}
        if stems != dat_stems:
            raise RuntimeError(f"PTB-XL .hea/.dat stem mismatch in {directory}")
        for name in names:
            jobs.append((urllib.parse.urljoin(url, name), root / "ptb-xl" / directory.rstrip("/") / name))
    if len(jobs) != PTB_COUNT * 2:
        raise RuntimeError(f"PTB-XL exposed {len(jobs)} waveform/header files; expected {PTB_COUNT * 2}")
    return jobs


def _external_jobs(root: Path, *, workers: int) -> list[tuple[str, Path]]:
    jobs: list[tuple[str, Path]] = []
    for source, expected in EXTERNAL_COUNTS.items():
        source_url = f"{CHALLENGE_BASE}/{source}/"
        groups = _subdirs(source_url, _GROUP_RE)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            listings = list(executor.map(lambda group: _files(urllib.parse.urljoin(source_url, group), (".hea", ".mat")), groups))
        source_jobs: list[tuple[str, Path]] = []
        stems: set[str] = set()
        mat_stems: set[str] = set()
        for group, names in zip(groups, listings, strict=True):
            group_url = urllib.parse.urljoin(source_url, group)
            group_name = group.rstrip("/")
            for name in names:
                if name.endswith(".hea"):
                    stems.add(Path(name).stem)
                else:
                    mat_stems.add(Path(name).stem)
                source_jobs.append((urllib.parse.urljoin(group_url, name), root / source / group_name / name))
        if stems != mat_stems or len(stems) != expected:
            raise RuntimeError(f"{source} .hea/.mat inventory mismatch: records={len(stems)}, expected={expected}")
        jobs.extend(source_jobs)
    return jobs


def _content_total(response: http.client.HTTPResponse, offset: int) -> int | None:
    content_range = response.getheader("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1]
        return None if total == "*" else int(total)
    length = response.getheader("Content-Length")
    return None if length is None else offset + int(length)


def _download_streaming(job: tuple[str, Path], *, attempts: int = 5) -> int:
    url, destination = job
    if destination.is_file() and destination.stat().st_size > 0:
        return destination.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            offset = partial.stat().st_size if partial.is_file() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else None
            response = _request(url, headers=headers, timeout=120.0)
            if offset and response.status == 200:
                partial.unlink(missing_ok=True)
                offset = 0
                mode = "wb"
            elif offset and response.status == 206:
                mode = "ab"
            elif not offset and response.status == 200:
                mode = "wb"
            else:
                response.read()
                raise RuntimeError(f"Unexpected HTTP {response.status} for {url}")
            total = _content_total(response, offset)
            with partial.open(mode) as handle:
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
            observed = partial.stat().st_size
            if total is not None and observed != total:
                raise RuntimeError(f"Incomplete download for {url}: {observed}/{total} bytes")
            partial.replace(destination)
            return destination.stat().st_size
        except (OSError, TimeoutError, http.client.HTTPException, RuntimeError) as exc:
            last_error = exc
            _drop_connection()
            if attempt + 1 < attempts:
                time.sleep(min(20.0, 2.0 * (attempt + 1)))
    raise RuntimeError(f"Failed waveform download after {attempts} attempts: {url}") from last_error


def _download_all(jobs: list[tuple[str, Path]], *, workers: int, rounds: int) -> tuple[int, int, list[int]]:
    pending = list(jobs)
    completed: dict[tuple[str, Path], int] = {}
    failures_by_round: list[int] = []
    for round_index in range(rounds + 1):
        failures: list[tuple[str, Path]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_download_streaming, job): job for job in pending}
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    completed[job] = int(future.result())
                except Exception:  # noqa: BLE001 - bounded batch recovery boundary
                    failures.append(job)
        failures_by_round.append(len(failures))
        pending = failures
        if not pending:
            break
        if round_index < rounds:
            time.sleep(min(30.0, 5.0 * (round_index + 1)))
    if pending:
        raise RuntimeError(f"Waveform corpus remains incomplete after recovery rounds: {len(pending)} files")
    return len(completed), sum(completed.values()), failures_by_round


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="trust_ecg_primary_data")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--recovery-rounds", type=int, default=4)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("--workers must be between 1 and 8 for waveform transfers")
    if not 0 <= args.recovery_rounds <= 8:
        raise SystemExit("--recovery-rounds must be between 0 and 8")
    root = Path(args.output_root).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve() if args.summary else root / "waveform_download_summary.json"
    if args.dry_run:
        print(json.dumps({
            "study": "TRUST-ECG",
            "protocol_version": "0.4.0",
            "development": {"original_ptbxl_v1_0_1": PTB_COUNT},
            "external": EXTERNAL_COUNTS,
            "challenge_ptbxl_waveforms": "prohibited_not_downloaded",
            "transport": "streaming_atomic_resume_with_http_range_and_batch_recovery",
            "workers": args.workers,
            "recovery_rounds": args.recovery_rounds,
        }, indent=2, sort_keys=True))
        return 0

    root.mkdir(parents=True, exist_ok=True)
    metadata = root / "ptb-xl" / "ptbxl_database.csv"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    if not metadata.is_file() or metadata.stat().st_size == 0:
        metadata.write_bytes(_read_small(PTB_METADATA_URL))
    ptb_jobs = _ptb_jobs(root)
    external_jobs = _external_jobs(root, workers=args.workers)
    total_files, total_bytes, failures = _download_all(
        ptb_jobs + external_jobs,
        workers=args.workers,
        rounds=args.recovery_rounds,
    )
    expected_files = PTB_COUNT * 2 + 2 * sum(EXTERNAL_COUNTS.values())
    if total_files != expected_files:
        raise RuntimeError(f"Completed file count {total_files} != expected {expected_files}")
    if list(root.rglob("*.partial")):
        raise RuntimeError("Partial waveform files remain after a supposedly complete download")
    summary = {
        "study": "TRUST-ECG",
        "protocol_version": "0.4.0",
        "development_source": "original_ptbxl_v1_0_1",
        "challenge_ptbxl_model_input": False,
        "ptbxl_records": PTB_COUNT,
        "external_records": EXTERNAL_COUNTS,
        "completed_files": total_files,
        "downloaded_or_reused_bytes": total_bytes,
        "failed_counts_by_round": failures,
        "ready_for_waveform_audit": True,
        "primary_data_root": str(root),
        "ptbxl_metadata": str(metadata),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
