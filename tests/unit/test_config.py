from pathlib import Path

import pytest

from speech_representations.config import ConfigError, load_config


def test_project_config_is_complete() -> None:
    config = load_config()
    assert config.root == Path.cwd()
    assert set(config.models) == {"wavlm_large", "hubert_large", "whisper_large_v3"}
    assert set(config.tasks) == {"speaker", "pitch", "noise", "emotion"}
    assert config.model("whisper_large_v3").representation_layers == 33
    assert config.task("pitch").kind == "regression"
    assert config.analysis.cka_max_samples == 500


def test_unknown_model_and_task_are_rejected() -> None:
    config = load_config()
    with pytest.raises(ConfigError, match="Unknown model"):
        config.model("missing")
    with pytest.raises(ConfigError, match="Unknown task"):
        config.task("missing")
