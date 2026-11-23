"""Audio loading and numerically stable noise mixing."""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

SAMPLE_RATE = 16_000


def load_audio(
    path: str | Path, sample_rate: int = SAMPLE_RATE, normalize: bool = True
) -> torch.Tensor:
    waveform, source_rate = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    waveform = waveform.squeeze(0).float()
    if normalize:
        std = waveform.std(unbiased=False)
        waveform = (waveform - waveform.mean()) / std.clamp_min(1e-8)
    return waveform


def match_length(waveform: torch.Tensor, length: int, start: int = 0) -> torch.Tensor:
    if waveform.numel() == 0:
        raise ValueError("Cannot resize an empty waveform")
    if waveform.numel() >= length:
        maximum_start = waveform.numel() - length
        offset = min(max(0, start), maximum_start)
        return waveform[offset : offset + length]
    repeats = (length + waveform.numel() - 1) // waveform.numel()
    return waveform.repeat(repeats)[:length]


def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    if clean.shape != noise.shape:
        raise ValueError(f"Waveform shapes must match, got {clean.shape} and {noise.shape}")
    clean_rms = clean.square().mean().sqrt().clamp_min(1e-8)
    noise_rms = noise.square().mean().sqrt().clamp_min(1e-8)
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    return clean + noise * (target_noise_rms / noise_rms)


def measured_snr_db(clean: torch.Tensor, mixed: torch.Tensor) -> float:
    residual = mixed - clean
    ratio = clean.square().mean().clamp_min(1e-12) / residual.square().mean().clamp_min(1e-12)
    return float(10.0 * torch.log10(ratio))
