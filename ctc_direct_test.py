import torch

from src.transforms.spec_augs import (
    FrequencyMasking,
    TimeMasking,
)

torch.manual_seed(42)

# имитируем MelSpectrogram:
# [channels, n_mels, time]
spectrogram = torch.rand(1, 128, 300)

freq_mask = FrequencyMasking(
    freq_mask_param=8,
)

time_mask = TimeMasking(
    time_mask_param=20,
)

masked = freq_mask(spectrogram.clone())
masked = time_mask(masked)

print("original shape:", spectrogram.shape)
print("masked shape:  ", masked.shape)

print(
    "changed values:",
    torch.sum(spectrogram != masked).item(),
)

print(
    "original zeros:",
    torch.sum(spectrogram == 0).item(),
)

print(
    "masked zeros:",
    torch.sum(masked == 0).item(),
)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 5))

plt.imshow(
    masked[0].numpy(),
    aspect="auto",
    origin="lower",
)

plt.xlabel("Time")
plt.ylabel("Mel bins")
plt.title("Spectrogram after FrequencyMasking + TimeMasking")

plt.show()