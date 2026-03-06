"""
HuggingFace Dataset Builder for Hindi ASR.

Builds a ``datasets.DatasetDict`` from a list of :class:`AudioSegment`
objects or from a directory of pre-processed audio files + transcript
manifest. The dataset is saved in Arrow format for efficient random-access
during Whisper fine-tuning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

_AUDIO_COLUMN = "audio"
_TRANSCRIPT_COLUMN = "sentence"
_SAMPLE_RATE = 16_000


def _segment_to_dict(segment) -> dict:
    """Convert an :class:`AudioSegment` to a plain dict for ``datasets``."""
    return {
        _AUDIO_COLUMN: {
            "array": segment.audio_array.tolist(),
            "sampling_rate": segment.sample_rate,
            "path": segment.source_path or "",
        },
        _TRANSCRIPT_COLUMN: segment.transcript,
    }


def build_dataset_from_segments(
    segments,
    test_size: float = 0.1,
    seed: int = 42,
) -> "datasets.DatasetDict":
    """
    Create a train/test ``DatasetDict`` from a list of :class:`AudioSegment`.

    Parameters
    ----------
    segments:
        List of :class:`~src.data.audio_segmentation.AudioSegment` objects.
    test_size:
        Fraction of data held out for the test split.
    seed:
        Random seed used for the train/test split.

    Returns
    -------
    ``datasets.DatasetDict`` with ``"train"`` and ``"test"`` keys.
    """
    try:
        from datasets import Dataset, DatasetDict, Audio
    except ImportError as exc:
        raise ImportError(
            "The `datasets` library is required. Install with: pip install datasets"
        ) from exc

    records = [_segment_to_dict(seg) for seg in segments]
    if not records:
        raise ValueError("No segments provided — cannot build an empty dataset.")

    ds = Dataset.from_list(records)
    ds = ds.cast_column(_AUDIO_COLUMN, Audio(sampling_rate=_SAMPLE_RATE))

    split = ds.train_test_split(test_size=test_size, seed=seed)
    logger.info(
        "Dataset built: %d train / %d test samples.",
        len(split["train"]),
        len(split["test"]),
    )
    return DatasetDict({"train": split["train"], "test": split["test"]})


def build_dataset_from_manifest(
    manifest_path: Union[str, Path],
    audio_root: Union[str, Path],
    test_size: float = 0.1,
    seed: int = 42,
) -> "datasets.DatasetDict":
    """
    Build a dataset from a JSONL manifest file.

    Each line of the manifest must be a JSON object with at minimum:
    ``{"audio_path": "relative/path.wav", "transcript": "..."}``

    Parameters
    ----------
    manifest_path:
        Path to the JSONL manifest file.
    audio_root:
        Root directory relative to which ``audio_path`` entries are resolved.
    test_size:
        Fraction of data used for the test split.
    seed:
        Random seed for the split.

    Returns
    -------
    ``datasets.DatasetDict`` with ``"train"`` and ``"test"`` keys.
    """
    try:
        from datasets import Dataset, DatasetDict, Audio
    except ImportError as exc:
        raise ImportError(
            "The `datasets` library is required. Install with: pip install datasets"
        ) from exc

    audio_root = Path(audio_root)
    manifest_path = Path(manifest_path)

    records: List[dict] = []
    missing: List[str] = []

    with open(manifest_path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", line_no, exc)
                continue

            audio_rel = entry.get("audio_path", "")
            transcript = entry.get("transcript", "").strip()
            audio_path = audio_root / audio_rel

            if not audio_path.exists():
                logger.warning("Audio file not found: %s", audio_path)
                missing.append(str(audio_path))
                continue

            if not transcript:
                logger.warning("Empty transcript for %s — skipping.", audio_path)
                continue

            records.append({
                _AUDIO_COLUMN: str(audio_path),
                _TRANSCRIPT_COLUMN: transcript,
            })

    if missing:
        logger.warning("%d audio files were missing and skipped.", len(missing))

    if not records:
        raise ValueError("No valid records found in manifest.")

    ds = Dataset.from_list(records)
    ds = ds.cast_column(_AUDIO_COLUMN, Audio(sampling_rate=_SAMPLE_RATE))
    split = ds.train_test_split(test_size=test_size, seed=seed)

    logger.info(
        "Manifest dataset built: %d train / %d test samples.",
        len(split["train"]),
        len(split["test"]),
    )
    return DatasetDict({"train": split["train"], "test": split["test"]})


def save_dataset(
    dataset: "datasets.DatasetDict",
    output_dir: Union[str, Path],
) -> None:
    """
    Save a ``DatasetDict`` to disk in Arrow / Parquet format.

    Parameters
    ----------
    dataset:
        The dataset to save.
    output_dir:
        Destination directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    logger.info("Dataset saved to %s", output_dir)


def load_dataset_from_disk(path: Union[str, Path]) -> "datasets.DatasetDict":
    """
    Load a previously saved :class:`datasets.DatasetDict` from disk.

    Parameters
    ----------
    path:
        Path produced by :func:`save_dataset`.

    Returns
    -------
    Loaded ``DatasetDict``.
    """
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise ImportError(
            "The `datasets` library is required. Install with: pip install datasets"
        ) from exc

    ds = load_from_disk(str(path))
    logger.info("Dataset loaded from %s", path)
    return ds
