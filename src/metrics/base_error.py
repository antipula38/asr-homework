from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer, calc_cer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved

class BaseErrorRateMetric(BaseMetric):
    def __init__(self, text_encoder, calc_metric, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.calc_metric = calc_metric

    def _decode(self, log_probs: Tensor, length: int) -> str:
        raise NotImplementedError

    def __call__(
            self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
        ):
        errors = []

        lengths = log_probs_length.detach().cpu().tolist()

        for log_prob_vec, length, target_text in zip(log_probs, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self._decode(log_prob_vec, length)
            errors.append(self.calc_metric(target_text, pred_text))

        return sum(errors) / len(errors)
    

class ArgmaxErrorMetric(BaseErrorRateMetric):
    def _decode(self, log_probs: Tensor, length: int) -> str:
        log_prob_vec = torch.argmax(log_probs[:length], dim=-1).detach().cpu().tolist()
        return self.text_encoder.ctc_decode(log_prob_vec)

class BeamSearchErrorMetric(BaseErrorRateMetric):
    def __init__(self, text_encoder, calc_metric, beam_size=10, *args, **kwargs):
        super().__init__(text_encoder, calc_metric, *args, **kwargs)
        self.beam_size = beam_size
    
    def _decode(self, log_probs: Tensor, length: int) -> str:
        return self.text_encoder.ctc_beam_search(log_probs[:length], beam_size=self.beam_size)

