import pytest
import torch

from speech_representations.data.audio import match_length, measured_snr_db, mix_at_snr


def test_mix_at_requested_snr() -> None:
    generator = torch.Generator().manual_seed(7)
    clean = torch.randn(16_000, generator=generator)
    noise = torch.randn(16_000, generator=generator)
    mixed = mix_at_snr(clean, noise, 10.0)
    assert measured_snr_db(clean, mixed) == pytest.approx(10.0, abs=1e-4)


def test_match_length_crops_and_loops() -> None:
    assert match_length(torch.arange(10), 4, start=3).tolist() == [3, 4, 5, 6]
    assert match_length(torch.tensor([1, 2, 3]), 7).tolist() == [1, 2, 3, 1, 2, 3, 1]
