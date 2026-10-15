from pathlib import Path

import torch

from src.trainer.inferencer import Inferencer
from src.utils.io_utils import ROOT_PATH


class EMAInferencer(Inferencer):
    """
    Inferencer that loads EMA model weights from a checkpoint.
    """

    def _from_pretrained(self, pretrained_path):
        checkpoint_path = Path(pretrained_path)

        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT_PATH / checkpoint_path

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if "ema_state_dict" not in checkpoint:
            raise KeyError(
                f"Checkpoint {checkpoint_path} does not contain "
                "'ema_state_dict'."
            )

        self.model.load_state_dict(checkpoint["ema_state_dict"])

        print(f"Loaded EMA weights from: {checkpoint_path}")