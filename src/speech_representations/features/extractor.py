"""Model-aware extraction of pooled hidden-layer representations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from transformers import HubertModel, WavLMModel, WhisperFeatureExtractor, WhisperModel

from speech_representations.config import ModelConfig


class SpeechFeatureExtractor:
    """Frozen Hugging Face encoder with correct padding-aware pooling."""

    def __init__(
        self,
        model_config: ModelConfig,
        device: torch.device,
        precision: str,
    ) -> None:
        self.config = model_config
        self.device = device
        self.precision = precision if device.type == "cuda" else "fp32"
        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]
        load_options = {
            "revision": model_config.revision,
            "dtype": dtype,
        }
        if model_config.name == "wavlm_large":
            self.model = WavLMModel.from_pretrained(model_config.hub_id, **load_options)
            self.preprocessor = None
        elif model_config.name == "hubert_large":
            self.model = HubertModel.from_pretrained(model_config.hub_id, **load_options)
            self.preprocessor = None
        elif model_config.name == "whisper_large_v3":
            self.model = WhisperModel.from_pretrained(model_config.hub_id, **load_options).encoder
            self.preprocessor = WhisperFeatureExtractor.from_pretrained(
                model_config.hub_id, revision=model_config.revision
            )
        else:  # guarded by typed project config, retained for direct API use
            raise ValueError(f"Unsupported model: {model_config.name}")
        self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def extract(self, waveforms: Sequence[torch.Tensor]) -> np.ndarray:
        """Return float32 pooled features with shape ``(batch, layers, hidden)``."""
        if not waveforms:
            raise ValueError("Cannot extract an empty batch")
        lengths = torch.tensor([waveform.numel() for waveform in waveforms], dtype=torch.long)
        # WavLM/HuBERT can accumulate meaningful fp16 drift when examples with
        # different padded lengths share a forward pass. Preserve scientific
        # invariance by only batching equal-length waveform-encoder inputs.
        if (
            self.config.architecture != "whisper"
            and len(waveforms) > 1
            and int(lengths.min()) != int(lengths.max())
        ):
            return np.stack([self.extract([waveform])[0] for waveform in waveforms])
        if self.config.architecture == "whisper":
            hidden_states, frame_mask = self._forward_whisper(waveforms, lengths)
        else:
            hidden_states, frame_mask = self._forward_waveform_encoder(waveforms, lengths)

        if len(hidden_states) != self.config.representation_layers:
            raise RuntimeError(
                f"{self.config.name} returned {len(hidden_states)} states; "
                f"expected {self.config.representation_layers}"
            )
        mask = frame_mask.unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1)
        pooled = [(state.float() * mask).sum(dim=1) / denominator for state in hidden_states]
        features = torch.stack(pooled, dim=1).cpu().numpy().astype(np.float32, copy=False)
        if features.shape[2] != self.config.hidden_size or not np.isfinite(features).all():
            raise RuntimeError(f"Invalid features produced by {self.config.name}: {features.shape}")
        return features

    def _forward_waveform_encoder(
        self, waveforms: Sequence[torch.Tensor], lengths: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        maximum = int(lengths.max())
        batch = torch.zeros((len(waveforms), maximum), dtype=torch.float32)
        sample_mask = torch.zeros_like(batch, dtype=torch.long)
        for index, waveform in enumerate(waveforms):
            length = waveform.numel()
            batch[index, :length] = waveform
            sample_mask[index, :length] = 1
        parameter_dtype = next(self.model.parameters()).dtype
        output = self.model(
            input_values=batch.to(self.device, dtype=parameter_dtype),
            attention_mask=sample_mask.to(self.device),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = output.hidden_states
        output_length = hidden_states[0].shape[1]
        frame_lengths = self.model._get_feat_extract_output_lengths(lengths.to(self.device))
        positions = torch.arange(output_length, device=self.device).unsqueeze(0)
        frame_mask = positions < frame_lengths.unsqueeze(1)
        return hidden_states, frame_mask

    def _forward_whisper(
        self, waveforms: Sequence[torch.Tensor], lengths: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        assert self.preprocessor is not None
        arrays = [waveform.cpu().numpy() for waveform in waveforms]
        inputs = self.preprocessor(
            arrays,
            sampling_rate=16_000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        parameter_dtype = next(self.model.parameters()).dtype
        output = self.model(
            input_features=inputs.input_features.to(self.device, dtype=parameter_dtype),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = output.hidden_states
        output_length = hidden_states[0].shape[1]
        # Whisper's encoder emits one vector per 320 input samples (20 ms), capped at 1500.
        frame_lengths = (
            torch.div(lengths + 319, 320, rounding_mode="floor")
            .clamp(min=1, max=output_length)
            .to(self.device)
        )
        positions = torch.arange(output_length, device=self.device).unsqueeze(0)
        return hidden_states, positions < frame_lengths.unsqueeze(1)
