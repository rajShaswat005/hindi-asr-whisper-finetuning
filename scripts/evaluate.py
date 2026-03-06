#!/usr/bin/env python3
"""
Evaluation Script — Compute WER on FLEURS Hindi Test Set.

Usage
-----
    python scripts/evaluate.py \\
        --model_path ./whisper-small-hindi \\
        --batch_size 8

The script:
1. Downloads the FLEURS Hindi test split.
2. Runs inference with the fine-tuned Whisper model.
3. Computes and prints Word Error Rate (WER) and Character Error Rate (CER).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow importing from src/ without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.wer_evaluator import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Whisper model on FLEURS Hindi")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned Whisper model directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory for the FLEURS dataset",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save evaluation results as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Evaluating model: %s", args.model_path)
    results = evaluate_model(
        model_path=args.model_path,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )

    print("\n" + "=" * 40)
    print(f"  WER : {results['wer']:.4f}  ({results['wer'] * 100:.2f}%)")
    print(f"  CER : {results['cer']:.4f}  ({results['cer'] * 100:.2f}%)")
    print("=" * 40 + "\n")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
