from pathlib import Path
import torch_audiomentations
from torch import Tensor, nn


class RIR(nn.Module):
    def __init__(self, ir_paths, *args, **kwargs):
        super().__init__()
        ir_paths = Path(ir_paths)
        wav_paths = [path for path in ir_paths.rglob("*.wav") if "_noise_" not in path.name.lower()]

        if len(wav_paths) == 0:
            raise ValueError(f"No .wav files found in {ir_paths}")
        self._aug = torch_audiomentations.ApplyImpulseResponse(ir_paths=wav_paths, *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
    