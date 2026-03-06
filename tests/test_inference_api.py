"""
Tests for the FastAPI inference API.

Uses TestClient from httpx (via starlette.testclient) to exercise the
/health and /transcribe endpoints without starting a real server.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def client():
    """Create a TestClient that patches model loading."""
    from fastapi.testclient import TestClient
    from api.inference_api import create_app
    import api.inference_api as api_module

    # Patch startup to avoid actually loading Whisper
    app = create_app(model_path="/fake/model/path")

    # Override on_event("startup") — just skip model loading in tests
    app.router.on_startup.clear()

    # Inject mock model + processor into module globals
    mock_processor = MagicMock()
    mock_processor.return_value.input_features = MagicMock()

    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([MagicMock()])

    api_module._model = mock_model
    api_module._processor = mock_processor
    api_module._device = "cpu"

    return TestClient(app)


class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok_status(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"


class TestTranscribeEndpoint:
    def test_missing_file_returns_422(self, client):
        resp = client.post("/transcribe")
        assert resp.status_code == 422

    def test_empty_file_returns_400(self, client):
        resp = client.post(
            "/transcribe",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert resp.status_code == 400

    def test_successful_transcription(self, client):
        import api.inference_api as api_module

        # Patch the transcription helper so we don't need a real audio file
        with patch.object(
            api_module,
            "_transcribe_audio_bytes",
            return_value="नमस्ते दुनिया",
        ):
            # Minimal WAV-like bytes (non-empty)
            fake_audio = b"RIFF" + b"\x00" * 36
            resp = client.post(
                "/transcribe",
                files={"file": ("test.wav", fake_audio, "audio/wav")},
            )

        assert resp.status_code == 200
        assert resp.json()["transcript"] == "नमस्ते दुनिया"

    def test_model_not_loaded_returns_503(self, client):
        import api.inference_api as api_module

        # Temporarily unset the model
        original_model = api_module._model
        api_module._model = None
        try:
            fake_audio = b"RIFF" + b"\x00" * 36
            resp = client.post(
                "/transcribe",
                files={"file": ("test.wav", fake_audio, "audio/wav")},
            )
            assert resp.status_code == 503
        finally:
            api_module._model = original_model
