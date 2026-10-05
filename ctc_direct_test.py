import torch

from src.model.deepspeech2 import DeepSpeech2


model = DeepSpeech2(
    n_feats=80,
    n_tokens=28,
)

x = torch.randn(2, 80, 501)
lengths = torch.tensor([501, 400])

result = model(
    spectrogram=x,
    spectrogram_length=lengths
)

print(result["log_probs"].shape)
print(result["log_probs_length"])