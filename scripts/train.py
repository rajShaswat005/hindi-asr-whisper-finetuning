#!/usr/bin/env python3
"""
Training Script — Fine-tune Whisper-Small on Hindi ASR.

Usage
-----
    python scripts/train.py \\
        --config configs/config.yaml \\
        --manifest data/train_manifest.jsonl \\
        --audio_root data/audio \\
        --output_dir ./whisper-small-hindi

The script:
1. Validates all audio URLs in the manifest (if URLs are present).
2. Builds a HuggingFace DatasetDict from the manifest.
3. Fine-tunes Whisper-Small using the supplied configuration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow importing from src/ without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.config import load_config
from training.trainer import WhisperFineTuner
from dataset.dataset_builder import build_dataset_from_manifest, load_dataset_from_disk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper-Small on Hindi ASR")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML training configuration file",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to JSONL manifest file (for building dataset from scratch)",
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        default=".",
        help="Root directory for audio files referenced in the manifest",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Path to a pre-built Arrow dataset (skips manifest processing)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs from config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    overrides = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.epochs is not None:
        overrides["num_train_epochs"] = args.epochs

    config = load_config(args.config if Path(args.config).exists() else None, **overrides)
    logger.info("Training configuration loaded: model=%s", config.model_name)

    # Build or load dataset
    if args.dataset_dir and Path(args.dataset_dir).exists():
        logger.info("Loading pre-built dataset from %s", args.dataset_dir)
        dataset = load_dataset_from_disk(args.dataset_dir)
    elif args.manifest:
        logger.info("Building dataset from manifest: %s", args.manifest)
        dataset = build_dataset_from_manifest(
            manifest_path=args.manifest,
            audio_root=args.audio_root,
        )
    else:
        logger.error(
            "Please provide either --manifest or --dataset_dir. "
            "Run `python scripts/train.py --help` for usage."
        )
        sys.exit(1)

    logger.info("Dataset: %s", dataset)

    # Fine-tune
    trainer = WhisperFineTuner(config)
    trainer.train(dataset)


if __name__ == "__main__":
    main()
