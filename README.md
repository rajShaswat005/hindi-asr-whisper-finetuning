# Hindi ASR Fine-Tuning with Whisper-Small

Production-grade Automatic Speech Recognition pipeline for Hindi speech using OpenAI Whisper.

## Project Overview

End-to-end Hindi ASR training pipeline including:

- Dataset ingestion and URL reconstruction
- Audio segmentation and transcript normalization
- HuggingFace dataset creation
- Whisper-small fine-tuning
- Word Error Rate evaluation on FLEURS Hindi
- FastAPI inference prototype

## Architecture

Pipeline stages:

1. **Data Retrieval** — URL validation, retry + caching logic, missing URL tracking
2. **Audio Segmentation** — Splits long recordings into ≤30 s clips aligned with transcripts
3. **Text Normalization** — Cleans Hindi (Devanagari) transcripts (NFC, invisible chars, digit conversion)
4. **Dataset Creation** — Builds a `datasets.DatasetDict` in Arrow format
5. **Whisper Fine-Tuning** — `Seq2SeqTrainer` wrapping `openai/whisper-small`
6. **Evaluation (WER)** — WER + CER on the FLEURS Hindi test set
7. **Inference API** — FastAPI server with `/transcribe` and `/health` endpoints

## Project Structure

```
hindi-asr-whisper-finetuning/
├── configs/
│   └── config.yaml            # Training hyperparameters
├── scripts/
│   ├── train.py               # Fine-tuning entry point
│   └── evaluate.py            # WER evaluation entry point
├── src/
│   ├── api/
│   │   └── inference_api.py   # FastAPI inference server
│   ├── data/
│   │   ├── audio_segmentation.py  # Audio loading & segmentation
│   │   ├── text_normalization.py  # Hindi text normalization
│   │   └── url_validator.py       # URL validation & caching
│   ├── dataset/
│   │   └── dataset_builder.py # HuggingFace DatasetDict builder
│   ├── evaluation/
│   │   └── wer_evaluator.py   # WER / CER on FLEURS Hindi
│   └── training/
│       ├── config.py          # Training configuration dataclass
│       └── trainer.py         # Whisper Seq2Seq fine-tuner
├── tests/                     # Pytest test suite (70 tests)
├── requirements.txt
└── setup.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Fine-Tuning

Prepare a JSONL manifest where each line is:
```json
{"audio_path": "relative/path/to/audio.wav", "transcript": "हिंदी पाठ"}
```

Then run:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --manifest data/train_manifest.jsonl \
  --audio_root data/audio \
  --output_dir ./whisper-small-hindi
```

### Evaluation (WER on FLEURS Hindi)

```bash
python scripts/evaluate.py \
  --model_path ./whisper-small-hindi \
  --batch_size 8 \
  --output_json results/eval.json
```

### Inference API

```bash
uvicorn src.api.inference_api:app --host 0.0.0.0 --port 8000
```

Send audio files to the `/transcribe` endpoint:

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" | python -m json.tool
```

Health check:

```bash
curl http://localhost:8000/health
```

## Dataset Issues

Some dataset URLs referenced in the assignment returned **404 / NoSuchKey errors**, indicating missing cloud objects.

The pipeline handles this via:

- **URL validation** — HEAD requests before downloading
- **Retry + caching** — exponential back-off, local Arrow cache
- **Missing URL tracking** — unavailable URLs logged to `missing_urls.jsonl`

This ensures training remains reproducible even when some data entries are unavailable.

## Model

| Property | Value |
|----------|-------|
| Base model | `openai/whisper-small` |
| Training objective | Sequence-to-sequence speech recognition |
| Evaluation metric | Word Error Rate (WER) |
| Language | Hindi (`hi`) |
| Sample rate | 16 kHz |
| Max context | 30 seconds |

## Running Tests

```bash
pytest tests/ -v
```

