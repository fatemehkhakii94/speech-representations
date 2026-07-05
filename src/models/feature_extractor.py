"""
Unified feature extractor for speech foundation models.

Provides a single interface to extract layer-wise hidden states from:
- WavLM-Large (self-supervised + denoising)
- HuBERT-Large (self-supervised, masked prediction)
- Whisper-Large-v3 encoder (supervised ASR)

All models are FROZEN — no gradients, no weight updates.
This module is used only for inference/feature extraction.
"""

import torch
import torch.nn as nn
from transformers import (
    WavLMModel,
    HubertModel,
    WhisperModel,
    WhisperFeatureExtractor,
)
from typing import List, Tuple, Optional

from src.config import ModelConfig, get_model_config


class FeatureExtractor(nn.Module):
    """
    Unified interface for extracting hidden states from speech models.
    
    Usage:
        extractor = FeatureExtractor.from_name("wavlm_large", device="cuda")
        hidden_states = extractor(waveform)
        # hidden_states: list of (batch, num_frames, hidden_dim) tensors
        # len(hidden_states) == num_layers + 1 (includes input embedding)
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        device: str = "cuda",
        preprocessor=None,
    ):
        super().__init__()
        self.model = model
        self.model_config = model_config
        self.device = device
        self.preprocessor = preprocessor
        
        # Freeze all parameters — we never train the backbone
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    @classmethod
    def from_name(cls, model_name: str, device: str = "cuda") -> "FeatureExtractor":
        """
        Factory method: create extractor from model name.
        
        Args:
            model_name: One of "wavlm_large", "hubert_large", "whisper_large_v3"
            device: "cuda" or "cpu"
        
        Returns:
            Configured FeatureExtractor instance
        """
        config = get_model_config(model_name)
        
        # Use safetensors format to avoid torch.load security restriction
        load_kwargs = {"use_safetensors": True}
        
        if model_name == "wavlm_large":
            model = WavLMModel.from_pretrained(config.hub_id, **load_kwargs)
            preprocessor = None
            
        elif model_name == "hubert_large":
            model = HubertModel.from_pretrained(config.hub_id, **load_kwargs)
            preprocessor = None
            
        elif model_name == "whisper_large_v3":
            model = WhisperModel.from_pretrained(config.hub_id, **load_kwargs)
            preprocessor = WhisperFeatureExtractor.from_pretrained(config.hub_id)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Move to device and optionally use fp16
        if config.use_fp16 and device == "cuda":
            model = model.half()
        model = model.to(device)
        
        return cls(
            model=model,
            model_config=config,
            device=device,
            preprocessor=preprocessor,
        )
    
    @property
    def num_layers(self) -> int:
        """Number of transformer layers (excludes input embedding layer)."""
        return self.model_config.num_layers
    
    @property
    def hidden_dim(self) -> int:
        """Hidden state dimensionality."""
        return self.model_config.hidden_dim
    
    @property
    def name(self) -> str:
        """Model name identifier."""
        return self.model_config.name
    
    @torch.no_grad()
    def forward(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
    ) -> List[torch.Tensor]:
        """
        Extract hidden states from all layers.
        
        Args:
            waveform: Input audio tensor
                      Shape: (batch_size, num_samples)
            sample_rate: Audio sample rate (must be 16000)
        
        Returns:
            List of tensors, one per layer (including input embedding as layer 0).
            Each tensor has shape: (batch_size, num_frames, hidden_dim)
            Total length: num_layers + 1
        """
        assert sample_rate == 16000, f"Expected 16kHz, got {sample_rate}Hz"
        
        if self.model_config.name == "whisper_large_v3":
            return self._forward_whisper(waveform)
        else:
            return self._forward_wav2vec_family(waveform)
    
    def _forward_wav2vec_family(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass for WavLM and HuBERT (wav2vec2-family models).
        
        Input: raw waveform (batch, samples)
        These models have an internal CNN feature encoder followed by transformer layers.
        """
        if self.model_config.use_fp16:
            waveform = waveform.half()
        waveform = waveform.to(self.device)
        
        outputs = self.model(
            input_values=waveform,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Index 0 = CNN feature encoder output (before any transformer)
        # Index 1..N = transformer layer outputs
        hidden_states = [h.float() for h in outputs.hidden_states]
        
        return hidden_states
    
    def _forward_whisper(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass for Whisper encoder.
        
        Input: raw waveform (batch, samples)
        Whisper requires mel spectrogram preprocessing before the encoder.
        """
        waveform_np = waveform.cpu().numpy()
        
        mel_features = self.preprocessor(
            waveform_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=480000,
        )
        
        input_features = mel_features.input_features
        
        if self.model_config.use_fp16:
            input_features = input_features.half()
        input_features = input_features.to(self.device)
        
        encoder_outputs = self.model.encoder(
            input_features=input_features,
            output_hidden_states=True,
            return_dict=True,
        )
        
        hidden_states = [h.float() for h in encoder_outputs.hidden_states]
        
        return hidden_states
    
    def get_num_frames(self, num_samples: int) -> int:
        """
        Calculate expected number of output frames for a given input length.
        
        For WavLM/HuBERT: ~320 samples per frame (20ms at 16kHz, ~50 fps)
        For Whisper: Fixed 1500 frames for any input up to 30s
        """
        if self.model_config.name == "whisper_large_v3":
            return 1500
        else:
            return num_samples // 320
    
    def __repr__(self) -> str:
        return (
            f"FeatureExtractor(\n"
            f"  model={self.model_config.name},\n"
            f"  layers={self.num_layers},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  device={self.device},\n"
            f"  fp16={self.model_config.use_fp16}\n"
            f")"
        )
