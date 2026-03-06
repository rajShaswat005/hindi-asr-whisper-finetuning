"""
Tests for URL validation and caching logic.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is on the path when running tests directly
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.url_validator import (
    _log_missing_url,
    _url_to_cache_path,
    batch_validate_urls,
    download_audio,
    load_missing_urls,
    reconstruct_urls,
    validate_url,
)


class TestReconstructUrls:
    def test_basic_reconstruction(self):
        base = "https://example.com/dataset/"
        paths = ["audio/001.wav", "audio/002.wav"]
        urls = reconstruct_urls(base, paths)
        assert urls == [
            "https://example.com/dataset/audio/001.wav",
            "https://example.com/dataset/audio/002.wav",
        ]

    def test_base_without_trailing_slash(self):
        base = "https://example.com/dataset"
        paths = ["/audio/001.wav"]
        urls = reconstruct_urls(base, paths)
        assert urls == ["https://example.com/dataset/audio/001.wav"]

    def test_empty_paths(self):
        assert reconstruct_urls("https://example.com/", []) == []


class TestUrlCachePath:
    def test_deterministic(self):
        url = "https://example.com/audio/test.wav"
        cache_dir = Path("/tmp/cache")
        p1 = _url_to_cache_path(url, cache_dir)
        p2 = _url_to_cache_path(url, cache_dir)
        assert p1 == p2

    def test_different_urls_different_paths(self):
        cache_dir = Path("/tmp/cache")
        p1 = _url_to_cache_path("https://a.com/1.wav", cache_dir)
        p2 = _url_to_cache_path("https://a.com/2.wav", cache_dir)
        assert p1 != p2

    def test_preserves_suffix(self):
        cache_dir = Path("/tmp/cache")
        p = _url_to_cache_path("https://example.com/audio/test.wav", cache_dir)
        assert p.suffix == ".wav"


class TestValidateUrl:
    def test_valid_url_returns_true(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_session = MagicMock()
        mock_session.head.return_value = mock_resp

        is_valid, code = validate_url("https://example.com/file.wav", session=mock_session)
        assert is_valid is True
        assert code == 200

    def test_404_returns_false(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_session = MagicMock()
        mock_session.head.return_value = mock_resp

        is_valid, code = validate_url("https://example.com/missing.wav", session=mock_session)
        assert is_valid is False
        assert code == 404

    def test_connection_error_returns_false(self):
        import requests as req
        mock_session = MagicMock()
        mock_session.head.side_effect = req.exceptions.ConnectionError("refused")

        is_valid, code = validate_url("https://example.com/file.wav", session=mock_session)
        assert is_valid is False
        assert code == 0


class TestDownloadAudio:
    def test_cache_hit_returns_existing_file(self, tmp_path):
        url = "https://example.com/audio/test.wav"
        cache_path = _url_to_cache_path(url, tmp_path)
        cache_path.write_bytes(b"RIFF_fake_wav_data")

        result = download_audio(url, cache_dir=tmp_path)
        assert result == cache_path

    def test_successful_download(self, tmp_path):
        url = "https://example.com/audio/new.wav"
        fake_content = b"RIFF_fake_wav_data_12345"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [fake_content]

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        result = download_audio(url, cache_dir=tmp_path, session=mock_session)
        assert result is not None
        assert result.exists()
        assert result.read_bytes() == fake_content

    def test_404_returns_none_and_logs(self, tmp_path):
        url = "https://example.com/missing.wav"
        log_path = tmp_path / "missing.jsonl"

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.reason = "Not Found"

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        result = download_audio(url, cache_dir=tmp_path, session=mock_session, missing_log=log_path)
        assert result is None
        assert log_path.exists()

        entries = load_missing_urls(log_path)
        assert len(entries) == 1
        assert entries[0]["url"] == url
        assert entries[0]["status_code"] == 404


class TestLogMissingUrl:
    def test_writes_jsonl_entry(self, tmp_path):
        log_path = tmp_path / "missing.jsonl"
        _log_missing_url("https://example.com/x.wav", 404, "Not Found", log_path)

        assert log_path.exists()
        with open(log_path) as fh:
            entry = json.loads(fh.readline())
        assert entry["url"] == "https://example.com/x.wav"
        assert entry["status_code"] == 404

    def test_appends_multiple_entries(self, tmp_path):
        log_path = tmp_path / "missing.jsonl"
        _log_missing_url("https://example.com/a.wav", 404, "Not Found", log_path)
        _log_missing_url("https://example.com/b.wav", 403, "Forbidden", log_path)

        entries = load_missing_urls(log_path)
        assert len(entries) == 2

    def test_no_log_path_does_not_raise(self):
        _log_missing_url("https://example.com/x.wav", 404, "Not Found", None)


class TestBatchValidateUrls:
    def test_all_valid(self):
        urls = ["https://example.com/a.wav", "https://example.com/b.wav"]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_session = MagicMock()
        mock_session.head.return_value = mock_resp

        results = batch_validate_urls(urls, session=mock_session)
        assert all(results.values())

    def test_mixed_validity(self, tmp_path):
        url_ok = "https://example.com/ok.wav"
        url_missing = "https://example.com/missing.wav"

        def head_side_effect(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200 if url == url_ok else 404
            return resp

        mock_session = MagicMock()
        mock_session.head.side_effect = head_side_effect

        log_path = tmp_path / "missing.jsonl"
        results = batch_validate_urls([url_ok, url_missing], session=mock_session, missing_log=log_path)

        assert results[url_ok] is True
        assert results[url_missing] is False
        entries = load_missing_urls(log_path)
        assert len(entries) == 1


class TestLoadMissingUrls:
    def test_empty_when_no_file(self, tmp_path):
        entries = load_missing_urls(tmp_path / "nonexistent.jsonl")
        assert entries == []

    def test_loads_all_entries(self, tmp_path):
        log_path = tmp_path / "missing.jsonl"
        _log_missing_url("https://a.com/x.wav", 404, "Not Found", log_path)
        _log_missing_url("https://b.com/y.wav", 500, "Error", log_path)

        entries = load_missing_urls(log_path)
        assert len(entries) == 2
        assert entries[0]["url"] == "https://a.com/x.wav"
        assert entries[1]["url"] == "https://b.com/y.wav"
