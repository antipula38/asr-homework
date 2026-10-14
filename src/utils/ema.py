import copy
import torch

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        self.model.eval()
        assert 0.0 <= decay < 1.0
        self.decay = decay
    


    @torch.no_grad()
    def update(self, model):
        ema_params = dict(self.model.named_parameters())
        model_params = dict(model.named_parameters())

        ema_buffers = dict(self.model.named_buffers())
        model_buffers = dict(model.named_buffers())

        for name, buf in model_buffers.items():
            ema_buf = ema_buffers[name]
            ema_buf.copy_(buf)

        for name, param in model_params.items():
            ema_param = ema_params[name]

            ema_param.mul_(self.decay)
            ema_param.add_(param, alpha=1 - self.decay)
