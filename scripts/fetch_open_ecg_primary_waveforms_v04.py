#!/usr/bin/env python3
"""Fetch the exact TRUST-ECG v0.4 primary waveform corpora, resumably and atomically.

Development: original PTB-XL v1.0.1 ``records500`` (.hea/.dat).
External: Challenge 2020 Georgia, CPSC2018 and CPSC2018-Extra (.hea/.mat).
Challenge-renamed PTB-XL waveforms are deliberately absent from this fetch plan.
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
PTB_METADATA_URL = "https://physionet.org/files/ptb-xl/1.0.1/ptbxl_database.csv"
CHALLENGE_BASE = "https://physionet.org/files/challenge-2020/1.0.2/training"
PTB_COUNT = 21837
EXTERNAL_COUNTS = {"georgia": 10344, "cpsc_2018": 6877, "cpsc_2018_extra": 3453}
_PTB_DIR_RE = re.compile(r"^\d{5}/$")
_GROUP_RE = re.compile(r"^g\d+/$")
_USER_AGENT = "TRUST-ECG-waveforms/0.4 (+https://github.com/AzizulHakim00/TRUST-ICU)"
_THREAD = threading.local()
_CHUNK_BYTES = 1024 * 1024


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
    if (
        parsed.scheme != "https"
        or parsed.hostname != "physionet.org"
        or parsed.username
        or parsed.password
    ):
        raise ValueError(f"Only https://physionet.org is allowed: {url!r}")
    return parsed


def _connection(timeout: float) -> http.client.HTTPSConnection:
    conn = getattr(_THREAD, "connection", None)
    if conn is None:
        conn = http.client.HTTPSConnection("physionet.org", timeout=timeout)
        _THREAD.connection = conn
    else:
        conn.timeout = timeout
    return conn


def _drop_connection() -> None:
    conn = getattr(_THREAD, "connection", None)
    if conn is not None:
        try:
            conn.close()
        finally:
            _THREAD.connection = None


def _request(url: str, *, headers: dict[str, str] | None = None, timeout: float = 120.0):
    parsed = _validate_url(url)
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    request_headers = {"User-Agent": _USER_AGENT, "Accept": "*/*"}
    if headers:
        request_headers.update(headers)
    conn = _connection(timeout)
    conn.request("GET", target, headers=request_headers)
    return conn.getresponse()


def _read_small(url: str, attempts: int = 5) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = _request(url, timeout=60.0)
            payload = response.read()
            if response.status != 200 or not payload:
                raise RuntimeError(f"HTTP {response.status} or empty response for {url}")
            return payload
        except (OSError, TimeoutError, http.client.HTTPException, RuntimeError) as exc:
            error = exc
            _drop_connection()
            if attempt + 1 < attempts:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"Failed to fetch {url}") from error


def _hrefs(url: str) -> list[str]:
    parser = _HrefParser()
    parser.feed(_read_small(url).decode("utf-8", errors="replace"))
    return parser.hrefs


def _subdirs(url: str, pattern: re.Pattern[str]) -> list[str]:
    values = []
    for href in _hrefs(url):
        parsed = urllib.parse.urlparse(href)
        if not parsed.scheme and not parsed.netloc and not parsed.query and pattern.fullmatch(parsed.path):
            values.append(parsed.path)
    result = sorted(set(values))
    if not result:
        raise RuntimeError(f"No expected subdirectories found at {url}")
    return result


def _files(url: str, suffixes: tuple[str, ...]) -> list[str]:
    result = []
    for href in _hrefs(url):
        parsed = urllib.parse.urlparse(href)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            continue
        name = Path(parsed.path).name
        if name and name not in {".", ".."} and name.endswith(suffixes):
            result.append(name)
    return sorted(set(result))


def _ptb_jobs(root: Path) -> list[tuple[str, Path]]:
    jobs: list[tuple[str, Path]] = []
    for directory in _subdirs(PTB_BASE, _PTB_DIR_RE):
        url = urllib.parse.urljoin(PTB_BASE, directory)
        names = _files(url, (".hea", ".dat"))
        hea = {Path(name).stem for name in names if name.endswith(".hea")}
        dat = {Path(name).stem for name in names if name.endswith(".dat")}
        if hea != dat:
            raise RuntimeError(f"PTB-XL header/waveform stem mismatch in {directory}")
        for name in names:
            destination = root / "ptb-xl" / "records500" / directory.rstrip("/") / name
            jobs.append((urllib.parse.urljoin(url, name), destination))
    if len(jobs) != PTB_COUNT * 2:
        raise RuntimeError(f"PTB-XL file inventory {len(jobs)} != {PTB_COUNT * 2}")
    return jobs


def _list_external_group(item: tuple[str, str]) -> tuple[str, list[str]]:
    source_url, group = item
    group_url = urllib.parse.urljoin(source_url, group)
    return group, _files(group_url, (".hea", ".mat"))


def _external_jobs(root: Path, workers: int) -> list[tuple[str, Path]]:
    all_jobs: list[tuple[str, Path]] = []
    for source, expected in EXTERNAL_COUNTS.items():
        source_url = f"{CHALLENGE_BASE}/{source}/"
        groups = _subdirs(source_url, _GROUP_RE)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            listings = list(executor.map(_list_external_group, [(source_url, group) for group in groups]))
        source_jobs: list[tuple[str, Path]] = []
        hea: set[str] = set()
        mat: set[str] = set()
        for group, names in listings:
            group_url = urllib.parse.urljoin(source_url, group)
            dirname = group.rstrip("/")
            for name in names:
                (hea if name.endswith(".hea") else mat).add(Path(name).stem)
                source_jobs.append(
                    (urllib.parse.urljoin(group_url, name), root / source / dirname / name)
                )
        if hea != mat or len(hea) != expected:
            raise RuntimeError(
                f"{source} inventory mismatch: headers={len(hea)}, mats={len(mat)}, expected={expected}"
            )
        all_jobs.extend(source_jobs)
    return all_jobs


def _response_total(response: http.client.HTTPResponse, offset: int) -> int | None:
    content_range = response.getheader("Content-Range")
    if content_range and "/" in content_range:
        value = content_range.rsplit("/", 1)[1]
        return None if value == "*" else int(value)
    content_length = response.getheader("Content-Length")
    return None if content_length is None else offset + int(content_length)


def _download(job: tuple[str, Path], attempts: int = 5) -> int:
    url, destination = job
    if destination.is_file() and destination.stat().st_size > 0:
        return destination.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            offset = partial.stat().st_size if partial.is_file() else 0
            response = _request(
                url,
                headers={"Range": f"bytes={offset}-"} if offset else None,
            )
            if offset and response.status == 206:
                mode = "ab"
            elif response.status == 200:
                if offset:
                    partial.unlink(missing_ok=True)
                    offset = 0
                mode = "wb"
            else:
                response.read()
                raise RuntimeError(f"Unexpected HTTP {response.status} for {url}")
            expected_total = _response_total(response, offset)
            with partial.open(mode) as handle:
                while True:
                    chunk = response.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
            observed = partial.stat().st_size
            if expected_total is not None and observed != expected_total:
                raise RuntimeError(f"Incomplete transfer {observed}/{expected_total} for {url}")
            partial.replace(destination)
            return destination.stat().st_size
        except (OSError, TimeoutError, http.client.HTTPException, RuntimeError) as exc:
            error = exc
            _drop_connection()
            if attempt + 1 < attempts:
                time.sleep(min(20.0, 2.0 * (attempt + 1)))
    raise RuntimeError(f"Failed after {attempts} attempts: {url}") from error


def _download_batches(
    jobs: list[tuple[str, Path]],
    *,
    workers: int,
    recovery_rounds: int,
) -> tuple[int, int, list[int]]:
    pending = list(jobs)
    completed: dict[tuple[str, Path], int] = {}
    failures_by_round: list[int] = []
    for round_index in range(recovery_rounds + 1):
        failures: list[tuple[str, Path]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_download, job): job for job in pending}
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    completed[job] = int(future.result())
                except Exception:  # noqa: BLE001 - bounded corpus-recovery boundary
                    failures.append(job)
        failures_by_round.append(len(failures))
        pending = failures
        if not pending:
            break
        if round_index < recovery_rounds:
            time.sleep(min(30.0, 5.0 * (round_index + 1)))
    if pending:
        raise RuntimeError(f"Waveform corpus incomplete after recovery: {len(pending)} files")
    if len(completed) != len(jobs):
        raise RuntimeError(f"Download accounting mismatch: {len(completed)} != {len(jobs)}")
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
        raise SystemExit("--workers must be between 1 and 8")
    if not 0 <= args.recovery_rounds <= 8:
        raise SystemExit("--recovery-rounds must be between 0 and 8")
    root = Path(args.output_root).expanduser().resolve()
    summary_path = (
        Path(args.summary).expanduser().resolve()
        if args.summary
        else root / "waveform_download_summary.json"
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "study": "TRUST-ECG",
                    "protocol_version": "0.4.0",
                    "development": {"original_ptbxl_v1_0_1": PTB_COUNT},
                    "external": EXTERNAL_COUNTS,
                    "challenge_ptbxl_waveforms": "prohibited_not_downloaded",
                    "ptb_destination_layout": "ptb-xl/records500/<group>/<record>.hea|.dat",
                    "transport": "streaming_atomic_http_range_resume_with_batch_recovery",
                    "workers": args.workers,
                    "recovery_rounds": args.recovery_rounds,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    root.mkdir(parents=True, exist_ok=True)
    metadata = root / "ptb-xl" / "ptbxl_database.csv"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    if not metadata.is_file() or metadata.stat().st_size == 0:
        metadata.write_bytes(_read_small(PTB_METADATA_URL))

    jobs = _ptb_jobs(root) + _external_jobs(root, args.workers)
    completed_files, total_bytes, failures = _download_batches(
        jobs,
        workers=args.workers,
        recovery_rounds=args.recovery_rounds,
    )
    expected_files = PTB_COUNT * 2 + 2 * sum(EXTERNAL_COUNTS.values())
    if completed_files != expected_files:
        raise RuntimeError(f"Completed file count {completed_files} != expected {expected_files}")
    if list(root.rglob("*.partial")):
        raise RuntimeError("Partial files remain after a supposedly complete corpus transfer")

    summary = {
        "study": "TRUST-ECG",
        "protocol_version": "0.4.0",
        "development_source": "original_ptbxl_v1_0_1",
        "challenge_ptbxl_model_input": False,
        "ptbxl_records": PTB_COUNT,
        "external_records": EXTERNAL_COUNTS,
        "completed_files": completed_files,
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
