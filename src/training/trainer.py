"""
Whisper Fine-Tuning Trainer.

Wraps the HuggingFace ``Seq2SeqTrainer`` to fine-tune ``openai/whisper-small``
on Hindi speech. Handles:
- Feature extractor + tokenizer preparation
- Data collation with padding
- WER metric computation during evaluation
- Checkpoint management
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .config import WhisperTrainingConfig

logger = logging.getLogger(__name__)

_AUDIO_COLUMN = "audio"
_TRANSCRIPT_COLUMN = "sentence"


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Collate a batch of audio features and label token IDs, applying padding.

    Parameters
    ----------
    processor:
        A ``WhisperProcessor`` instance (feature extractor + tokenizer).
    decoder_start_token_id:
        Token ID used to start decoder generation.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        # Separate inputs (audio) from labels (token ids)
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id with -100 so it is ignored in the loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove the BOS token if it was prepended by the tokenizer
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class WhisperFineTuner:
    """
    High-level trainer for Whisper-Small Hindi ASR fine-tuning.

    Parameters
    ----------
    config:
        :class:`~src.training.config.WhisperTrainingConfig` with all
        hyperparameters.
    """

    def __init__(self, config: WhisperTrainingConfig) -> None:
        self.config = config
        self._processor = None
        self._model = None

    @property
    def processor(self):
        """Lazily load the WhisperProcessor."""
        if self._processor is None:
            from transformers import WhisperProcessor

            self._processor = WhisperProcessor.from_pretrained(
                self.config.model_name,
                language=self.config.language,
                task=self.config.task,
            )
        return self._processor

    @property
    def model(self):
        """Lazily load the WhisperForConditionalGeneration model."""
        if self._model is None:
            from transformers import WhisperForConditionalGeneration

            self._model = WhisperForConditionalGeneration.from_pretrained(
                self.config.model_name
            )
            # Force the model to always predict in Hindi
            self._model.config.forced_decoder_ids = None
            self._model.config.suppress_tokens = []
        return self._model

    def prepare_dataset(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a single dataset example.

        Converts raw audio arrays to log-mel spectrogram input features and
        tokenizes the transcript.

        Parameters
        ----------
        batch:
            A single dataset example containing ``audio`` and ``sentence``.

        Returns
        -------
        Dict with ``input_features`` and ``labels`` keys.
        """
        audio = batch[_AUDIO_COLUMN]
        batch["input_features"] = self.processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="np",
        ).input_features[0]

        batch["labels"] = self.processor.tokenizer(
            batch[_TRANSCRIPT_COLUMN]
        ).input_ids
        return batch

    def _compute_metrics(self, pred) -> Dict[str, float]:
        """Compute WER for a batch of predictions."""
        import evaluate

        wer_metric = evaluate.load("wer")

        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with the pad token id for decoding
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

    def train(self, dataset: "datasets.DatasetDict") -> None:
        """
        Run the fine-tuning loop.

        Parameters
        ----------
        dataset:
            ``DatasetDict`` with ``"train"`` and ``"test"`` splits.
        """
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

        logger.info("Preprocessing dataset...")
        processed = dataset.map(
            self.prepare_dataset,
            remove_columns=dataset.column_names["train"],
            num_proc=1,
        )

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            warmup_steps=self.config.warmup_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            fp16=self.config.fp16,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            predict_with_generate=self.config.predict_with_generate,
            generation_max_length=self.config.generation_max_length,
            push_to_hub=self.config.push_to_hub,
            report_to=["none"],
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=processed["train"],
            eval_dataset=processed["test"],
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        logger.info("Starting training...")
        trainer.train()

        logger.info("Saving model to %s", self.config.output_dir)
        trainer.save_model(self.config.output_dir)
        self.processor.save_pretrained(self.config.output_dir)
        logger.info("Training complete.")
