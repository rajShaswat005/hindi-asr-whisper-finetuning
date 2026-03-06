"""
Audio Segmentation Module.

Splits long audio recordings into short clips aligned with transcript
segments, resamples to 16 kHz mono, and filters clips that exceed
Whisper's 30-second context window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000   # Hz required by Whisper
MAX_SEGMENT_DURATION = 30.0   # seconds — Whisper's hard context limit


@dataclass
class AudioSegment:
    """A single audio segment with its associated transcript."""

    audio_array: np.ndarray    # float32, shape (num_samples,)
    sample_rate: int
    transcript: str
    start_time: float          # seconds within original recording
    end_time: float            # seconds within original recording
    source_path: Optional[str] = field(default=None)

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end_time - self.start_time

    def is_valid(self) -> bool:
        """Return True when the segment passes basic quality checks."""
        return (
            self.duration > 0.1
            and self.duration <= MAX_SEGMENT_DURATION
            and len(self.audio_array) > 0
            and self.transcript.strip() != ""
        )


def load_audio(path: str, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, convert to mono float32 and resample.

    Parameters
    ----------
    path:
        Path to the audio file.
    target_sr:
        Target sample rate.

    Returns
    -------
    (audio_array, sample_rate)
    """
    try:
        import librosa
    except ImportError as exc:
        raise ImportError("librosa is required for audio loading. Install with: pip install librosa") from exc

    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32), sr


def segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    """
    Extract a time-bounded slice from a numpy audio array.

    Parameters
    ----------
    audio:
        Full audio waveform (float32).
    sample_rate:
        Sample rate of *audio*.
    start_time:
        Segment start in seconds.
    end_time:
        Segment end in seconds.

    Returns
    -------
    Audio slice as float32 numpy array.
    """
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    return audio[start_sample:end_sample].astype(np.float32)


def build_segments_from_manifest(
    audio_path: str,
    manifest: List[dict],
    target_sr: int = TARGET_SAMPLE_RATE,
) -> List[AudioSegment]:
    """
    Build a list of :class:`AudioSegment` objects from a manifest.

    Each manifest entry is expected to contain:
    ``{"start": float, "end": float, "transcript": str}``.

    Parameters
    ----------
    audio_path:
        Path to the source audio file.
    manifest:
        List of segment dictionaries.
    target_sr:
        Target sample rate.

    Returns
    -------
    List of valid :class:`AudioSegment` instances.
    """
    audio, sr = load_audio(audio_path, target_sr=target_sr)
    segments: List[AudioSegment] = []
    skipped = 0

    for entry in manifest:
        start = float(entry["start"])
        end = float(entry["end"])
        transcript = str(entry.get("transcript", "")).strip()

        if end - start <= 0 or end - start > MAX_SEGMENT_DURATION:
            logger.warning(
                "Skipping segment [%.2f, %.2f] from %s: duration out of range.",
                start, end, audio_path,
            )
            skipped += 1
            continue

        clip = segment_audio(audio, sr, start, end)
        seg = AudioSegment(
            audio_array=clip,
            sample_rate=sr,
            transcript=transcript,
            start_time=start,
            end_time=end,
            source_path=audio_path,
        )
        if seg.is_valid():
            segments.append(seg)
        else:
            skipped += 1

    logger.info(
        "Built %d segments from %s (%d skipped).",
        len(segments), audio_path, skipped,
    )
    return segments


def pad_or_trim_audio(
    audio: np.ndarray,
    target_length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Pad or trim audio to exactly *target_length* samples.

    Parameters
    ----------
    audio:
        1-D float32 waveform.
    target_length:
        Desired number of samples.
    pad_value:
        Value used for padding.

    Returns
    -------
    Audio array of length *target_length*.
    """
    if len(audio) >= target_length:
        return audio[:target_length]
    padding = np.full(target_length - len(audio), pad_value, dtype=np.float32)
    return np.concatenate([audio, padding])


def normalize_audio_amplitude(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalise peak amplitude of a waveform to *target_peak*.

    Parameters
    ----------
    audio:
        1-D float32 waveform.
    target_peak:
        Desired peak absolute value.

    Returns
    -------
    Normalised waveform.
    """
    peak = np.abs(audio).max()
    if peak == 0.0:
        return audio
    return (audio / peak * target_peak).astype(np.float32)
