#!/usr/bin/env python3
"""Fetch public metadata and WFDB headers needed for TRUST-ECG pre-waveform audits.

The v0.4 study develops directly on original PTB-XL and therefore does not require original
PTB-XL headers merely to reverse-map Challenge record names. Challenge filenames are discovered
from live PhysioNet listings. Waveform samples are never downloaded by this script.
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
from collections import Counter
from collections.abc import Iterable
from html.parser import HTMLParser
from pathlib import Path

CHALLENGE_BASE = "https://physionet.org/files/challenge-2020/1.0.2/training"
PTBXL_METADATA_URL = "https://physionet.org/files/ptb-xl/1.0.1/ptbxl_database.csv"
PTBXL_RECORDS500_URL = "https://physionet.org/files/ptb-xl/1.0.1/records500/"
EXPECTED_SOURCES = {
    "ptb-xl": 21837,
    "georgia": 10344,
    "cpsc_2018": 6877,
    "cpsc_2018_extra": 3453,
}
_USER_AGENT = "TRUST-ECG-header-audit/0.4 (+https://github.com/AzizulHakim00/TRUST-ICU)"
_GROUP_RE = re.compile(r"^g\d+/$")
_PTB_DIR_RE = re.compile(r"^\d{5}/$")
_THREAD_LOCAL = threading.local()


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


def _connection(*, timeout: float) -> http.client.HTTPSConnection:
    connection = getattr(_THREAD_LOCAL, "physionet_connection", None)
    if connection is None:
        connection = http.client.HTTPSConnection("physionet.org", timeout=timeout)
        _THREAD_LOCAL.physionet_connection = connection
    else:
        connection.timeout = timeout
    return connection


def _drop_connection() -> None:
    connection = getattr(_THREAD_LOCAL, "physionet_connection", None)
    if connection is not None:
        try:
            connection.close()
        finally:
            _THREAD_LOCAL.physionet_connection = None


def _read_url(url: str, *, attempts: int = 3, timeout: float = 30.0) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "physionet.org" or parsed.username or parsed.password:
        raise ValueError(f"TRUST-ECG header fetcher only permits https://physionet.org URLs: {url!r}")
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            connection = _connection(timeout=timeout)
            connection.request("GET", target, headers={"User-Agent": _USER_AGENT, "Accept": "*/*"})
            response = connection.getresponse()
            payload = response.read()
            if response.status != 200:
                raise RuntimeError(f"PhysioNet returned HTTP {response.status} for {url!r}")
            return payload
        except (http.client.HTTPException, OSError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            _drop_connection()
            if attempt + 1 == attempts:
                break
            time.sleep(0.75 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url!r} after {attempts} attempts") from last_error


def _directory_hrefs(url: str) -> list[str]:
    parser = _HrefParser()
    parser.feed(_read_url(url, attempts=5, timeout=45.0).decode("utf-8", errors="replace"))
    return parser.hrefs


def _safe_leaf(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"Unexpected absolute or decorated directory href: {value!r}")
    leaf = Path(parsed.path).name
    if not leaf or leaf in {".", ".."} or "/" in leaf or "\\" in leaf:
        raise ValueError(f"Unsafe directory entry: {value!r}")
    return leaf


def _discover_subdirectories(root_url: str, pattern: re.Pattern[str]) -> list[str]:
    entries: set[str] = set()
    for href in _directory_hrefs(root_url):
        parsed = urllib.parse.urlparse(href)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            continue
        candidate = parsed.path
        if pattern.fullmatch(candidate):
            entries.add(candidate)
    if not entries:
        raise RuntimeError(f"No expected subdirectories discovered at {root_url}")
    return sorted(entries)


def _discover_headers_in_directory(directory_url: str) -> list[str]:
    headers: set[str] = set()
    for href in _directory_hrefs(directory_url):
        parsed = urllib.parse.urlparse(href)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            continue
        if not parsed.path.endswith(".hea"):
            continue
        headers.add(_safe_leaf(parsed.path))
    return sorted(headers)


def _parallel_map(function, items: Iterable[str], *, workers: int):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, items))


def _download_one(job: tuple[str, Path]) -> tuple[str, int]:
    url, destination = job
    if destination.is_file() and destination.stat().st_size > 0:
        return str(destination), destination.stat().st_size
    payload = _read_url(url)
    if not payload:
        raise RuntimeError(f"Refusing to write empty download from {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".partial.{os.getpid()}.{threading.get_ident()}")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return str(destination), len(payload)


def _download_jobs(
    jobs: list[tuple[str, Path]],
    *,
    workers: int,
    recovery_rounds: int,
) -> tuple[int, int, list[int]]:
    pending = list(jobs)
    completed: dict[tuple[str, Path], int] = {}
    failed_counts: list[int] = []
    last_errors: dict[tuple[str, Path], str] = {}

    for round_index in range(recovery_rounds + 1):
        if not pending:
            break
        failures: list[tuple[str, Path]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_job = {executor.submit(_download_one, job): job for job in pending}
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    _, size = future.result()
                except Exception as exc:  # noqa: BLE001 - aggregate retry boundary
                    failures.append(job)
                    last_errors[job] = f"{type(exc).__name__}: {exc}"
                else:
                    completed[job] = int(size)
                    last_errors.pop(job, None)

        failed_counts.append(len(failures))
        if not failures:
            pending = []
            break
        pending = failures
        if round_index < recovery_rounds:
            _drop_connection()
            time.sleep(min(30.0, 5.0 * (round_index + 1)))

    if pending:
        sample = [
            {"url": job[0], "error": last_errors.get(job, "unknown_error")}
            for job in pending[:5]
        ]
        raise RuntimeError(
            "Header download remained incomplete after bounded recovery rounds: "
            f"failed={len(pending)}, sample={json.dumps(sample, sort_keys=True)}"
        )
    if len(completed) != len(jobs):
        raise RuntimeError(
            f"Header download accounting mismatch: completed={len(completed)}, expected={len(jobs)}"
        )
    return len(completed), sum(completed.values()), failed_counts


def _challenge_jobs(output_root: Path, *, workers: int) -> tuple[list[tuple[str, Path]], dict[str, int]]:
    challenge_root = output_root / "challenge"
    jobs: list[tuple[str, Path]] = []
    discovered_counts: dict[str, int] = {}
    for source, expected in EXPECTED_SOURCES.items():
        source_url = f"{CHALLENGE_BASE}/{source}/"
        groups = _discover_subdirectories(source_url, _GROUP_RE)
        group_urls = [urllib.parse.urljoin(source_url, group) for group in groups]
        listings = _parallel_map(_discover_headers_in_directory, group_urls, workers=workers)
        source_jobs: list[tuple[str, Path]] = []
        for group, group_url, filenames in zip(groups, group_urls, listings, strict=True):
            group_name = group.rstrip("/")
            for filename in filenames:
                source_jobs.append(
                    (
                        urllib.parse.urljoin(group_url, filename),
                        challenge_root / source / group_name / filename,
                    )
                )
        if len(source_jobs) != expected:
            raise RuntimeError(
                f"PhysioNet listing for {source} exposed {len(source_jobs)} headers; expected {expected}."
            )
        discovered_counts[source] = len(source_jobs)
        jobs.extend(source_jobs)
    return jobs, discovered_counts


def _ptbxl_original_jobs(output_root: Path, *, workers: int) -> list[tuple[str, Path]]:
    original_root = output_root / "ptbxl_original" / "records500"
    directories = _discover_subdirectories(PTBXL_RECORDS500_URL, _PTB_DIR_RE)
    directory_urls = [urllib.parse.urljoin(PTBXL_RECORDS500_URL, item) for item in directories]
    listings = _parallel_map(_discover_headers_in_directory, directory_urls, workers=workers)
    jobs: list[tuple[str, Path]] = []
    for directory, directory_url, filenames in zip(directories, directory_urls, listings, strict=True):
        dirname = directory.rstrip("/")
        for filename in filenames:
            jobs.append(
                (
                    urllib.parse.urljoin(directory_url, filename),
                    original_root / dirname / filename,
                )
            )
    if len(jobs) != EXPECTED_SOURCES["ptb-xl"]:
        raise RuntimeError(
            f"PTB-XL v1.0.1 records500 exposed {len(jobs)} headers; expected 21837."
        )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="open_ecg_header_only_data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--recovery-rounds", type=int, default=4)
    parser.add_argument("--summary", default=None)
    parser.add_argument(
        "--skip-original-ptbxl-headers",
        action="store_true",
        help="v0.4 mode: do not fetch records500 headers because reverse crosswalk is retired.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 32:
        raise SystemExit("--workers must be between 1 and 32")
    if not 0 <= args.recovery_rounds <= 8:
        raise SystemExit("--recovery-rounds must be between 0 and 8")
    output_root = Path(args.output_root).expanduser().resolve()
    summary_path = (
        Path(args.summary).expanduser().resolve()
        if args.summary
        else output_root / "header_download_summary.json"
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "study": "TRUST-ECG",
                    "stage": "real_header_only_public_data_fetch",
                    "challenge_version": "1.0.2",
                    "ptbxl_version": "1.0.1",
                    "sources": EXPECTED_SOURCES,
                    "downloads_waveforms": False,
                    "downloads_original_ptbxl_headers": not args.skip_original_ptbxl_headers,
                    "filename_discovery": "live_physionet_directory_listings",
                    "transport": "bounded_parallel_https_with_batch_failure_recovery",
                    "workers": args.workers,
                    "recovery_rounds": args.recovery_rounds,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "ptbxl_database.csv"
    metadata_path.write_bytes(_read_url(PTBXL_METADATA_URL, attempts=5, timeout=45.0))

    challenge_jobs, discovered = _challenge_jobs(output_root, workers=args.workers)
    challenge_downloaded, challenge_bytes, challenge_failures = _download_jobs(
        challenge_jobs,
        workers=args.workers,
        recovery_rounds=args.recovery_rounds,
    )

    original_downloaded = 0
    original_bytes = 0
    original_failures: list[int] = []
    if not args.skip_original_ptbxl_headers:
        original_jobs = _ptbxl_original_jobs(output_root, workers=args.workers)
        original_downloaded, original_bytes, original_failures = _download_jobs(
            original_jobs,
            workers=args.workers,
            recovery_rounds=args.recovery_rounds,
        )

    observed_challenge = Counter()
    for source in EXPECTED_SOURCES:
        observed_challenge[source] = len(list((output_root / "challenge" / source).rglob("*.hea")))
    if dict(observed_challenge) != EXPECTED_SOURCES:
        raise RuntimeError(f"Downloaded Challenge header counts are invalid: {dict(observed_challenge)}")
    observed_original = len(list((output_root / "ptbxl_original" / "records500").rglob("*.hea")))
    if not args.skip_original_ptbxl_headers and observed_original != EXPECTED_SOURCES["ptb-xl"]:
        raise RuntimeError(f"Downloaded original PTB-XL header count is {observed_original}, expected 21837")
    if args.skip_original_ptbxl_headers and observed_original != 0:
        raise RuntimeError("v0.4 header-only fetch unexpectedly created original PTB-XL headers")

    summary = {
        "study": "TRUST-ECG",
        "stage": "real_header_only_public_data_fetch",
        "challenge_version": "1.0.2",
        "ptbxl_version": "1.0.1",
        "filename_discovery": "live_physionet_directory_listings",
        "transport": "bounded_parallel_https_with_batch_failure_recovery",
        "workers": args.workers,
        "recovery_rounds": args.recovery_rounds,
        "waveform_files_downloaded": 0,
        "challenge_discovered_counts": discovered,
        "challenge_downloaded_counts": dict(observed_challenge),
        "challenge_header_files": challenge_downloaded,
        "challenge_header_bytes": challenge_bytes,
        "challenge_failed_counts_by_round": challenge_failures,
        "ptbxl_original_headers_requested": not args.skip_original_ptbxl_headers,
        "ptbxl_original_header_files": original_downloaded,
        "ptbxl_original_header_bytes": original_bytes,
        "ptbxl_original_failed_counts_by_round": original_failures,
        "ptbxl_metadata_bytes": metadata_path.stat().st_size,
        "ready_for_header_audit": True,
        "challenge_root": str(output_root / "challenge"),
        "ptbxl_metadata": str(metadata_path),
    }
    if not args.skip_original_ptbxl_headers:
        summary["ptbxl_original_root"] = str(output_root / "ptbxl_original")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
