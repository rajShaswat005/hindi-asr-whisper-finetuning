"""
FastAPI Inference Prototype for Hindi ASR.

Provides two endpoints:
- ``POST /transcribe``  — accepts an uploaded WAV/MP3/FLAC file and returns
  the Hindi transcript.
- ``GET  /health``      — liveness probe.

Usage
-----
    uvicorn src.api.inference_api:app --host 0.0.0.0 --port 8000

Or programmatically::

    import uvicorn
    from src.api.inference_api import create_app

    app = create_app(model_path="./whisper-small-hindi")
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded globals — populated on first request or at startup
# ---------------------------------------------------------------------------
_model = None
_processor = None
_device = None

_DEFAULT_MODEL_PATH = os.environ.get("ASR_MODEL_PATH", "./whisper-small-hindi")
_SAMPLE_RATE = 16_000
_MAX_DURATION_SECONDS = 30.0


def _load_model(model_path: str) -> None:
    """Load Whisper model and processor into module-level globals."""
    global _model, _processor, _device

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading Whisper processor from %s ...", model_path)
    _processor = WhisperProcessor.from_pretrained(model_path)

    logger.info("Loading Whisper model from %s ...", model_path)
    _model = WhisperForConditionalGeneration.from_pretrained(model_path).to(_device)
    _model.eval()

    logger.info("Model loaded on %s", _device)


def _transcribe_audio_bytes(audio_bytes: bytes, filename: str) -> str:
    """
    Transcribe raw audio bytes.

    Parameters
    ----------
    audio_bytes:
        Raw file bytes (WAV / MP3 / FLAC / etc.).
    filename:
        Original filename — used to infer the format.

    Returns
    -------
    Decoded Hindi transcript string.
    """
    import librosa
    import torch

    suffix = Path(filename).suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio, _ = librosa.load(tmp_path, sr=_SAMPLE_RATE, mono=True)
    finally:
        os.unlink(tmp_path)

    # Truncate to Whisper's context window
    max_samples = int(_MAX_DURATION_SECONDS * _SAMPLE_RATE)
    if len(audio) > max_samples:
        logger.warning(
            "Audio is longer than %ds — truncating.", _MAX_DURATION_SECONDS
        )
        audio = audio[:max_samples]

    inputs = _processor(
        audio,
        sampling_rate=_SAMPLE_RATE,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        generated_ids = _model.generate(
            inputs.input_features,
            language="hindi",
            task="transcribe",
        )

    transcript = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcript.strip()


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------

try:
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


def create_app(model_path: Optional[str] = None):
    """
    Create and configure the FastAPI application.

    Parameters
    ----------
    model_path:
        Path to the fine-tuned Whisper model directory.
        Defaults to the ``ASR_MODEL_PATH`` environment variable, or
        ``"./whisper-small-hindi"``.

    Returns
    -------
    Configured ``FastAPI`` instance.
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required. Install with: pip install fastapi uvicorn[standard]"
        )

    resolved_path = model_path or _DEFAULT_MODEL_PATH

    @asynccontextmanager
    async def lifespan(application: "FastAPI"):  # pragma: no cover
        _load_model(resolved_path)
        yield

    app = FastAPI(
        title="Hindi ASR – Whisper Inference API",
        description="Transcribe Hindi speech using a fine-tuned OpenAI Whisper-Small model.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", summary="Health check")
    async def health() -> JSONResponse:
        """Return service liveness status."""
        return JSONResponse({"status": "ok", "model": resolved_path})

    @app.post("/transcribe", summary="Transcribe Hindi audio")
    async def transcribe(file: UploadFile = File(...)) -> JSONResponse:
        """
        Transcribe an uploaded audio file to Hindi text.

        Accepts WAV, MP3, FLAC, OGG, and most other common audio formats.
        Maximum supported duration: 30 seconds.

        Returns
        -------
        JSON: ``{"transcript": "<decoded_text>"}``
        """
        if _model is None or _processor is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet. Try again shortly.")

        allowed_content_types = {
            "audio/wav",
            "audio/x-wav",
            "audio/mpeg",
            "audio/mp3",
            "audio/flac",
            "audio/ogg",
            "audio/webm",
            "application/octet-stream",
        }
        if file.content_type and file.content_type not in allowed_content_types:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported media type: {file.content_type}",
            )

        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty file received.")

        try:
            transcript = _transcribe_audio_bytes(audio_bytes, file.filename or "audio.wav")
        except Exception as exc:
            logger.exception("Transcription failed: %s", exc)
            raise HTTPException(status_code=500, detail="Transcription failed.") from exc

        return JSONResponse({"transcript": transcript})

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (used by ``uvicorn src.api.inference_api:app``)
# ---------------------------------------------------------------------------
app = create_app()
