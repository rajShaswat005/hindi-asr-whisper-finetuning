"""
Word Error Rate Evaluator.

Computes WER (and CER) for a fine-tuned Whisper model on the FLEURS Hindi
test set or any ``DatasetDict`` with an ``audio`` column and a reference
transcript column.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

_AUDIO_COLUMN = "audio"
_TRANSCRIPT_COLUMN = "transcription"  # FLEURS uses 'transcription'
_FLEURS_DATASET = "google/fleurs"
_FLEURS_LANGUAGE = "hi_in"


def load_fleurs_test(
    language: str = _FLEURS_LANGUAGE,
    cache_dir: Optional[str] = None,
) -> "datasets.Dataset":
    """
    Load the FLEURS Hindi test split.

    Parameters
    ----------
    language:
        FLEURS language code (default ``"hi_in"`` for Hindi).
    cache_dir:
        Optional cache directory for HuggingFace datasets.

    Returns
    -------
    ``datasets.Dataset`` with ``audio`` and ``transcription`` columns.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` library is required. Install with: pip install datasets"
        ) from exc

    ds = load_dataset(
        _FLEURS_DATASET,
        language,
        split="test",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    logger.info("Loaded FLEURS Hindi test set: %d examples", len(ds))
    return ds


def transcribe_batch(
    model,
    processor,
    audio_arrays: List,
    sampling_rates: List[int],
    batch_size: int = 8,
) -> List[str]:
    """
    Transcribe a list of audio arrays using a Whisper model.

    Parameters
    ----------
    model:
        A ``WhisperForConditionalGeneration`` instance.
    processor:
        A ``WhisperProcessor`` instance.
    audio_arrays:
        List of numpy float32 waveforms.
    sampling_rates:
        Sampling rate for each waveform (should all be 16 kHz).
    batch_size:
        Number of examples processed per forward pass.

    Returns
    -------
    List of decoded transcript strings.
    """
    import torch

    device = next(model.parameters()).device
    all_predictions: List[str] = []

    for start in range(0, len(audio_arrays), batch_size):
        batch_audio = audio_arrays[start:start + batch_size]
        batch_sr = sampling_rates[start:start + batch_size]

        inputs = processor(
            batch_audio,
            sampling_rate=batch_sr[0],
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_features,
                language="hindi",
                task="transcribe",
            )

        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        all_predictions.extend(decoded)

    return all_predictions


def compute_wer(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute Word Error Rate.

    Parameters
    ----------
    predictions:
        List of model-predicted transcript strings.
    references:
        List of ground-truth transcript strings.

    Returns
    -------
    WER as a float in [0, 1+].
    """
    try:
        import evaluate
    except ImportError as exc:
        raise ImportError(
            "The `evaluate` library is required. Install with: pip install evaluate"
        ) from exc

    wer_metric = evaluate.load("wer")
    return wer_metric.compute(predictions=predictions, references=references)


def compute_cer(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute Character Error Rate.

    Parameters
    ----------
    predictions:
        List of model-predicted transcript strings.
    references:
        List of ground-truth transcript strings.

    Returns
    -------
    CER as a float in [0, 1+].
    """
    try:
        import evaluate
    except ImportError as exc:
        raise ImportError(
            "The `evaluate` library is required. Install with: pip install evaluate"
        ) from exc

    cer_metric = evaluate.load("cer")
    return cer_metric.compute(predictions=predictions, references=references)


def evaluate_model(
    model_path: Union[str, Path],
    dataset: Optional["datasets.Dataset"] = None,
    batch_size: int = 8,
    cache_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a fine-tuned Whisper model and return WER / CER metrics.

    Parameters
    ----------
    model_path:
        Path to the directory containing the saved model + processor.
    dataset:
        Optional pre-loaded dataset. If ``None``, the FLEURS Hindi test
        set is downloaded automatically.
    batch_size:
        Batch size for inference.
    cache_dir:
        Optional HuggingFace cache directory.

    Returns
    -------
    ``{"wer": float, "cer": float}``
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import torch

    model_path = str(model_path)

    logger.info("Loading model from %s ...", model_path)
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Model loaded on %s", device)

    if dataset is None:
        dataset = load_fleurs_test(cache_dir=cache_dir)

    transcript_col = _TRANSCRIPT_COLUMN if _TRANSCRIPT_COLUMN in dataset.column_names else "sentence"

    audio_arrays = [ex["array"] for ex in dataset[_AUDIO_COLUMN]]
    sampling_rates = [ex["sampling_rate"] for ex in dataset[_AUDIO_COLUMN]]
    references = dataset[transcript_col]

    logger.info("Running inference on %d examples...", len(audio_arrays))
    predictions = transcribe_batch(
        model, processor, audio_arrays, sampling_rates, batch_size=batch_size
    )

    wer = compute_wer(predictions, references)
    cer = compute_cer(predictions, references)

    results = {"wer": round(wer, 4), "cer": round(cer, 4)}
    logger.info("Evaluation results: WER=%.4f  CER=%.4f", wer, cer)
    return results
