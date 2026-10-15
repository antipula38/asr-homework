import re
from string import ascii_lowercase
from collections import defaultdict
from pyctcdecode import build_ctcdecoder
import math
import heapq
import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = '^'

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.tensor([self.char2ind[char] for char in text], dtype=torch.long).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        EMPTY_IND = self.char2ind[self.EMPTY_TOK]
        last_ind = -1
        res = ""
        for ind in inds:
            if ind != last_ind and ind != EMPTY_IND:
                res += self.ind2char[int(ind)]
            last_ind = ind
        return res.strip()

    def ctc_beam_search(self, log_probs, beam_size=10) -> str:
        blank_ind = self.char2ind[self.EMPTY_TOK]
        neg_inf = -float("inf")
        log_probs = log_probs.detach().cpu().tolist()

        def logaddexp(a, b):
            if a == neg_inf:
                return b
            if b == neg_inf:
                return a

            m = max(a, b)
            return m + math.log(
                math.exp(a - m) + math.exp(b - m)
            )


        beam = {"": [0.0, neg_inf]}

        for log_prob_t in log_probs:
            next_beam = defaultdict(lambda: [neg_inf, neg_inf])
            for prefix, (p_b, p_nb) in beam.items():
                for ind, log_p in enumerate(log_prob_t):

                    if ind == blank_ind: # blank
                        next_beam[prefix][0] = logaddexp(
                            next_beam[prefix][0],
                            logaddexp(p_b, p_nb) + log_p
                        )
                        continue

                    char = self.ind2char[ind]

                    if len(prefix) > 0 and prefix[-1] == char:
                        next_beam[prefix][1] = logaddexp(
                            next_beam[prefix][1],
                            p_nb + log_p
                        ) # a -> a

                        new_prefix = prefix + char

                        next_beam[new_prefix][1] = logaddexp(
                            next_beam[new_prefix][1],
                            p_b + log_p
                        ) #a -> ^ -> a

                    else:
                        new_prefix = prefix + char

                        next_beam[new_prefix][1] = logaddexp(
                            next_beam[new_prefix][1],
                            logaddexp(p_b, p_nb) + log_p
                        )

            beam = dict(
                heapq.nlargest(
                    beam_size,
                    next_beam.items(),
                    key=lambda item: logaddexp(
                        item[1][0],
                        item[1][1]
                    )
                )
            )
        
        best_prefix, _ = max(
            beam.items(),
            key=lambda item: logaddexp(item[1][0], item[1][1])
        )

        return best_prefix.strip()

    def load_lm(self, lm_path, alpha=0.5, beta=1.0):

        labels = [
            "" if char == self.EMPTY_TOK else char.upper()
            for char in self.vocab
        ]

        self.lm_decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=str(lm_path),
            alpha=float(alpha),
            beta=float(beta),
        )

    def ctc_decode_lm(self, log_probs, beam_size=25):
        if not hasattr(self, "lm_decoder"):
            raise RuntimeError("Load LM before decoding")

        assert log_probs.ndim == 2
        assert log_probs.shape[-1] == len(self.vocab)

        probs = log_probs.exp().detach().cpu().numpy()

        return self.lm_decoder.decode(
            probs,
            beam_width=beam_size,
        ).strip().lower()

            


    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
