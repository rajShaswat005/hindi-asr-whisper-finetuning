"""
Tests for audio segmentation utilities.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.audio_segmentation import (
    MAX_SEGMENT_DURATION,
    TARGET_SAMPLE_RATE,
    AudioSegment,
    normalize_audio_amplitude,
    pad_or_trim_audio,
    segment_audio,
)


class TestAudioSegment:
    def _make_segment(self, duration: float = 2.0, transcript: str = "नमस्ते") -> AudioSegment:
        sr = TARGET_SAMPLE_RATE
        samples = int(duration * sr)
        audio = np.zeros(samples, dtype=np.float32)
        return AudioSegment(
            audio_array=audio,
            sample_rate=sr,
            transcript=transcript,
            start_time=0.0,
            end_time=duration,
        )

    def test_duration_property(self):
        seg = self._make_segment(duration=3.5)
        assert abs(seg.duration - 3.5) < 1e-6

    def test_is_valid_good_segment(self):
        seg = self._make_segment(duration=2.0, transcript="नमस्ते")
        assert seg.is_valid()

    def test_is_valid_empty_transcript(self):
        seg = self._make_segment(duration=2.0, transcript="   ")
        assert not seg.is_valid()

    def test_is_valid_too_long(self):
        seg = self._make_segment(duration=MAX_SEGMENT_DURATION + 0.1)
        assert not seg.is_valid()

    def test_is_valid_too_short(self):
        seg = self._make_segment(duration=0.05)
        assert not seg.is_valid()


class TestSegmentAudio:
    def test_extracts_correct_slice(self):
        sr = TARGET_SAMPLE_RATE
        audio = np.arange(sr * 5, dtype=np.float32)  # 5-second ramp

        clip = segment_audio(audio, sr, start_time=1.0, end_time=3.0)
        assert len(clip) == 2 * sr

    def test_returns_float32(self):
        sr = TARGET_SAMPLE_RATE
        audio = np.ones(sr * 2, dtype=np.float64)
        clip = segment_audio(audio, sr, start_time=0.0, end_time=1.0)
        assert clip.dtype == np.float32

    def test_full_audio(self):
        sr = TARGET_SAMPLE_RATE
        audio = np.ones(sr * 3, dtype=np.float32)
        clip = segment_audio(audio, sr, 0.0, 3.0)
        assert len(clip) == len(audio)


class TestPadOrTrimAudio:
    def test_trim_longer_audio(self):
        audio = np.ones(100, dtype=np.float32)
        result = pad_or_trim_audio(audio, 50)
        assert len(result) == 50

    def test_pad_shorter_audio(self):
        audio = np.ones(30, dtype=np.float32)
        result = pad_or_trim_audio(audio, 50)
        assert len(result) == 50
        assert result[30] == 0.0

    def test_exact_length_unchanged(self):
        audio = np.ones(50, dtype=np.float32)
        result = pad_or_trim_audio(audio, 50)
        assert len(result) == 50
        np.testing.assert_array_equal(result, audio)


class TestNormalizeAudioAmplitude:
    def test_peak_after_normalization(self):
        audio = np.array([0.1, 0.5, -1.0, 0.3], dtype=np.float32)
        normalized = normalize_audio_amplitude(audio, target_peak=0.95)
        assert abs(np.abs(normalized).max() - 0.95) < 1e-5

    def test_silent_audio_unchanged(self):
        audio = np.zeros(100, dtype=np.float32)
        result = normalize_audio_amplitude(audio)
        np.testing.assert_array_equal(result, audio)

    def test_returns_float32(self):
        audio = np.array([1.0, -1.0], dtype=np.float64)
        result = normalize_audio_amplitude(audio)
        assert result.dtype == np.float32
