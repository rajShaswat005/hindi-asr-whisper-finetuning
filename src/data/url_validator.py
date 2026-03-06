"""
URL Validator and Dataset Retriever.

Handles URL validation, retry logic with exponential back-off,
local caching, and tracking of missing / inaccessible URLs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path("./data_cache/audio")
_DEFAULT_MISSING_LOG = Path("./missing_urls.jsonl")
_CONNECT_TIMEOUT = 10   # seconds
_READ_TIMEOUT = 60      # seconds
_MAX_RETRIES = 3
_BACKOFF_FACTOR = 1.5   # exponential back-off multiplier


def _build_session(max_retries: int = _MAX_RETRIES, backoff_factor: float = _BACKOFF_FACTOR) -> requests.Session:
    """Create a requests Session with retry behaviour on transient errors."""
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _url_to_cache_path(url: str, cache_dir: Path) -> Path:
    """Derive a deterministic local file path from a URL."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix or ".bin"
    return cache_dir / f"{url_hash}{suffix}"


def validate_url(url: str, session: Optional[requests.Session] = None) -> Tuple[bool, int]:
    """
    Check whether a URL is accessible via an HTTP HEAD request.

    Parameters
    ----------
    url:
        The URL to check.
    session:
        Optional pre-built requests Session. A new one is created if omitted.

    Returns
    -------
    (is_valid, status_code)
        ``is_valid`` is True only when the status code is 2xx.
    """
    sess = session or _build_session()
    try:
        resp = sess.head(url, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT), allow_redirects=True)
        is_valid = resp.status_code < 400
        return is_valid, resp.status_code
    except requests.RequestException as exc:
        logger.warning("HEAD request failed for %s: %s", url, exc)
        return False, 0


def download_audio(
    url: str,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    session: Optional[requests.Session] = None,
    missing_log: Optional[Path] = None,
) -> Optional[Path]:
    """
    Download an audio file from *url*, caching it locally.

    Parameters
    ----------
    url:
        Direct link to the audio file.
    cache_dir:
        Directory where cached files are stored.
    session:
        Optional pre-built requests Session.
    missing_log:
        If provided, details of unavailable URLs are appended here as JSONL.

    Returns
    -------
    Path to the cached local file, or ``None`` when the URL is unavailable.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = _url_to_cache_path(url, cache_dir)
    if local_path.exists():
        logger.debug("Cache hit: %s -> %s", url, local_path)
        return local_path

    sess = session or _build_session()
    try:
        resp = sess.get(url, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT), stream=True)
        if resp.status_code >= 400:
            _log_missing_url(url, resp.status_code, str(resp.reason), missing_log)
            return None

        with open(local_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)

        logger.info("Downloaded %s -> %s", url, local_path)
        return local_path

    except requests.RequestException as exc:
        _log_missing_url(url, 0, str(exc), missing_log)
        return None


def _log_missing_url(
    url: str,
    status_code: int,
    reason: str,
    missing_log: Optional[Path],
) -> None:
    """Append a missing URL entry to the JSONL log file."""
    logger.warning("Unavailable URL (status=%d): %s — %s", status_code, url, reason)
    if missing_log is None:
        return
    log_path = Path(missing_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "url": url,
        "status_code": status_code,
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def reconstruct_urls(
    base_url: str,
    relative_paths: List[str],
) -> List[str]:
    """
    Reconstruct full URLs from a base URL and a list of relative paths.

    Parameters
    ----------
    base_url:
        The bucket / CDN root, e.g. ``"https://storage.example.com/dataset/"``.
    relative_paths:
        Relative object keys, e.g. ``["audio/001.wav", "audio/002.wav"]``.

    Returns
    -------
    List of fully-qualified URLs.
    """
    base = base_url.rstrip("/")
    return [f"{base}/{path.lstrip('/')}" for path in relative_paths]


def batch_validate_urls(
    urls: List[str],
    session: Optional[requests.Session] = None,
    missing_log: Optional[Path] = None,
) -> Dict[str, bool]:
    """
    Validate a batch of URLs and report which are accessible.

    Parameters
    ----------
    urls:
        List of URLs to validate.
    session:
        Optional shared requests Session.
    missing_log:
        Path to JSONL file for recording unavailable URLs.

    Returns
    -------
    Mapping of URL -> ``True`` (accessible) / ``False`` (unavailable).
    """
    sess = session or _build_session()
    results: Dict[str, bool] = {}
    for url in urls:
        is_valid, status_code = validate_url(url, session=sess)
        results[url] = is_valid
        if not is_valid:
            _log_missing_url(url, status_code, "Validation failed", missing_log)
    return results


def load_missing_urls(missing_log: Path) -> List[Dict]:
    """Load all entries previously written to the missing URLs JSONL log."""
    log_path = Path(missing_log)
    if not log_path.exists():
        return []
    entries: List[Dict] = []
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
