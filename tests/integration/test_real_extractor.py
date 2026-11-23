import pytest
import torch

from speech_representations.config import load_config
from speech_representations.features.extractor import SpeechFeatureExtractor


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("model_name", ["wavlm_large", "hubert_large", "whisper_large_v3"])
def test_real_extractor_is_batch_invariant(model_name: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = load_config()
    extractor = SpeechFeatureExtractor(config.model(model_name), torch.device("cuda"), "fp16")
    generator = torch.Generator().manual_seed(3)
    target = torch.randn(24_000, generator=generator)
    longer = torch.randn(48_000, generator=generator)
    alone = extractor.extract([target])[0]
    batched = extractor.extract([target, longer])[0]
    torch.testing.assert_close(
        torch.from_numpy(alone), torch.from_numpy(batched), rtol=3e-3, atol=3e-3
    )
