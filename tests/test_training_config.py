"""
Tests for training configuration.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.config import WhisperTrainingConfig, load_config


class TestWhisperTrainingConfig:
    def test_default_values(self):
        config = WhisperTrainingConfig()
        assert config.model_name == "openai/whisper-small"
        assert config.language == "hindi"
        assert config.task == "transcribe"
        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 16
        assert config.sample_rate == 16_000

    def test_custom_values(self):
        config = WhisperTrainingConfig(
            model_name="openai/whisper-medium",
            num_train_epochs=5,
            per_device_train_batch_size=8,
        )
        assert config.model_name == "openai/whisper-medium"
        assert config.num_train_epochs == 5
        assert config.per_device_train_batch_size == 8


class TestLoadConfig:
    def test_defaults_without_yaml(self):
        config = load_config(yaml_path=None)
        assert isinstance(config, WhisperTrainingConfig)
        assert config.model_name == "openai/whisper-small"

    def test_overrides_applied(self):
        config = load_config(yaml_path=None, num_train_epochs=10, output_dir="/tmp/test")
        assert config.num_train_epochs == 10
        assert config.output_dir == "/tmp/test"

    def test_invalid_override_raises(self):
        with pytest.raises(ValueError, match="Unknown configuration key"):
            load_config(yaml_path=None, nonexistent_key="value")

    def test_load_from_yaml(self, tmp_path):
        yaml_content = """\
model:
  name: openai/whisper-small
  language: hindi
  task: transcribe

training:
  num_train_epochs: 7
  per_device_train_batch_size: 4
  learning_rate: 5.0e-5
"""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml_content)

        config = load_config(yaml_path=yaml_path)
        assert config.num_train_epochs == 7
        assert config.per_device_train_batch_size == 4
        assert abs(config.learning_rate - 5e-5) < 1e-10

    def test_override_beats_yaml(self, tmp_path):
        yaml_content = """\
training:
  num_train_epochs: 7
"""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml_content)

        config = load_config(yaml_path=yaml_path, num_train_epochs=99)
        assert config.num_train_epochs == 99
