"""
Configuration system for speech representation experiments.

Design:
- YAML files define experiments
- This module loads, validates, and provides access to config values
- Every experiment parameter lives in config, not in code
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration for a speech foundation model."""
    name: str                          # e.g., "wavlm_large"
    hub_id: str                        # HuggingFace model ID
    num_layers: int                    # Number of transformer layers
    hidden_dim: int                    # Hidden state dimensionality
    use_fp16: bool = True              # Use half precision for extraction


@dataclass
class DataConfig:
    """Configuration for dataset and preprocessing."""
    dataset_name: str                  # e.g., "librispeech", "cremad"
    data_dir: str                      # Path to raw data
    sample_rate: int = 16000           # Target sample rate
    max_duration: float = 10.0         # Max audio duration in seconds
    min_duration: float = 1.0          # Min audio duration in seconds


@dataclass
class ExtractionConfig:
    """Configuration for feature extraction."""
    model: ModelConfig
    data: DataConfig
    output_dir: str                    # Where to save features
    batch_size: int = 4                # Batch size for extraction
    num_workers: int = 4               # DataLoader workers


@dataclass
class ProbeConfig:
    """Configuration for probing experiments."""
    probe_type: str = "linear"         # "linear" or "mlp"
    hidden_dim: int = 256              # MLP hidden dimension (ignored for linear)
    dropout: float = 0.1              # Dropout rate (MLP only)


@dataclass
class TrainingConfig:
    """Configuration for probe training."""
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 5                  # Early stopping patience
    num_workers: int = 4
    seed: int = 42


@dataclass
class ProbingExperimentConfig:
    """Full configuration for a probing experiment."""
    # What we're probing
    property_name: str                 # e.g., "phoneme", "speaker", "emotion"
    num_classes: int                   # Number of output classes
    level: str                         # "frame" or "utterance"

    # Which model and layers
    model_name: str                    # e.g., "wavlm_large"
    features_dir: str                  # Path to cached features
    layers: Optional[list] = None      # Which layers to probe (None = all)

    # Probe setup
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Data
    pooling: str = "mean"              # "mean", "max" (for utterance-level)

    # Output
    results_dir: str = "results/probing"


def load_yaml(path: str) -> dict:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


# =============================================================================
# Predefined model configurations (these don't change)
# =============================================================================

MODELS = {
    "wavlm_large": ModelConfig(
        name="wavlm_large",
        hub_id="microsoft/wavlm-large",
        num_layers=24,
        hidden_dim=1024,
    ),
    "hubert_large": ModelConfig(
        name="hubert_large",
        hub_id="facebook/hubert-large-ll60k",
        num_layers=24,
        hidden_dim=1024,
    ),
    "whisper_large_v3": ModelConfig(
        name="whisper_large_v3",
        hub_id="openai/whisper-large-v3",
        num_layers=32,
        hidden_dim=1280,
    ),
}


def get_model_config(name: str) -> ModelConfig:
    """Get predefined model configuration by name."""
    if name not in MODELS:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODELS.keys())}"
        )
    return MODELS[name]
