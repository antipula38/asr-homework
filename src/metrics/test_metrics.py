import torch
from src.metrics.utils import *
from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from src.metrics.cer import ArgmaxCERMetric
from src.metrics.wer import ArgmaxWERMetric

encoder = CTCTextEncoder()

path = [0, 3, 3, 0, 21, 0, 20, 20, 0]

B = 1
T = len(path)
C = len(encoder)
log_probs = torch.full((B, T, C), -10.0)
log_probs[0, torch.arange(len(path)), path] = 10.0
predictions = torch.argmax(log_probs, dim=-1)
print(predictions)

log_probs_length = torch.tensor([len(path)])
text = ["cut and dog"]
cer_metric = ArgmaxCERMetric(text_encoder=encoder)
wer_metric = ArgmaxWERMetric(text_encoder=encoder)

cer = cer_metric(
    log_probs=log_probs,
    log_probs_length=log_probs_length,
    text=text,
)

wer = wer_metric(
    log_probs=log_probs,
    log_probs_length=log_probs_length,
    text=text,
)

print("CER:", cer)
print("WER:", wer)