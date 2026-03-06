"""
Training Configuration for Whisper Fine-Tuning.

Provides a dataclass-based configuration with sensible defaults, plus
a loader that reads from a YAML file and overrides with keyword arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class WhisperTrainingConfig:
    """Hyperparameters and paths used during Whisper fine-tuning."""

    # Model
    model_name: str = "openai/whisper-small"
    language: str = "hindi"
    task: str = "transcribe"

    # Output
    output_dir: str = "./whisper-small-hindi"

    # Training loop
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 8
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    lr_scheduler_type: str = "linear"
    fp16: bool = True

    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 25
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False

    # Generation
    predict_with_generate: bool = True
    generation_max_length: int = 225

    # HuggingFace Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

    # Data
    max_audio_length_seconds: float = 30.0
    sample_rate: int = 16_000
    cache_dir: str = "./data_cache"
    missing_urls_log: str = "./missing_urls.jsonl"


def load_config(
    yaml_path: Optional[Union[str, Path]] = None,
    **overrides: Any,
) -> WhisperTrainingConfig:
    """
    Load training configuration from a YAML file with optional overrides.

    Parameters
    ----------
    yaml_path:
        Path to a YAML configuration file.  If ``None``, default values
        from :class:`WhisperTrainingConfig` are used.
    **overrides:
        Keyword arguments that override values loaded from the YAML file.

    Returns
    -------
    Populated :class:`WhisperTrainingConfig`.
    """
    config = WhisperTrainingConfig()

    if yaml_path is not None:
        import yaml  # lazy import — only needed when a file is given

        with open(yaml_path, encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        # Flatten nested sections
        merged: Dict[str, Any] = {}
        for section in raw.values():
            if isinstance(section, dict):
                merged.update(section)

        for key, value in merged.items():
            if hasattr(config, key):
                setattr(config, key, value)

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: '{key}'")

    return config
