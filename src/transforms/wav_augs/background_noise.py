import torch_audiomentations
from pathlib import Path
from torch import Tensor, nn


class AddBackgroundNoise(nn.Module):
    def __init__(self, background_paths, *args, **kwargs):
        super().__init__()
        background_paths = Path(background_paths)
        wav_paths = list(background_paths.rglob("*.wav"))

        if len(wav_paths) == 0:
            raise ValueError(f"No .wav files found in {background_paths}")
        self._aug = torch_audiomentations.AddBackgroundNoise(background_paths=background_paths, *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
