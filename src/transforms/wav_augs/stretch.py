import librosa
import torch
import random
from torch import Tensor, nn


class TimeStretch(nn.Module):
    def __init__(self, min_rate=0.9, max_rate=1.1, p=0.3):
        super().__init__()
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.p = p

    def __call__(self, data: Tensor):
        if random.random() > self.p:
            return data 
        rate = random.uniform(self.min_rate, self.max_rate)
        x = data.squeeze(0).detach().cpu().numpy()
        aug = librosa.effects.time_stretch(x, rate=rate)
        return torch.from_numpy(aug).unsqueeze(0)