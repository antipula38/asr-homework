from src.text_encoder.ctc_text_encoder import CTCTextEncoder
import torch
from itertools import product
from collections import defaultdict


def brute_force_ctc(encoder, probs):
    scores = defaultdict(float)

    T, C = probs.shape

    for path in product(range(C), repeat=T):
        path_prob = 1.0

        for t, token in enumerate(path):
            path_prob *= probs[t, token].item()

        text = encoder.ctc_decode(path)
        scores[text] += path_prob

    return max(scores.items(), key=lambda x: x[1])
encoder = CTCTextEncoder(alphabet=["a", "b"])

probs = torch.tensor([
    [0.15, 0.55, 0.30],  # t1
    [0.55, 0.35, 0.10],  # t2
    [0.10, 0.60, 0.30],  # t3
    [0.30, 0.20, 0.50],  # t4
])

log_probs = torch.log(probs)

print("greedy:", encoder.ctc_decode(log_probs.argmax(dim=-1)))
print("beam:", encoder.ctc_beam_search(log_probs, beam_size=20))
exact_text, exact_prob = brute_force_ctc(encoder, probs)

beam_text = encoder.ctc_beam_search(
    torch.log(probs),
    beam_size=100
)

greedy_text = encoder.ctc_decode(
    probs.argmax(dim=-1)
)

print("greedy:", greedy_text)
print("beam:", beam_text)
print("exact:", exact_text, exact_prob)

print( beam_text == exact_text)