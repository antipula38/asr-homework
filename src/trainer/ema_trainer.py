from src.trainer.trainer import Trainer
from src.utils.ema import ModelEMA
import torch


class EMATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        decay = self.cfg_trainer.get("ema_decay", 0.999)
        self.ema = ModelEMA(self.model, decay=decay)

        if getattr(self, "_pending_ema_state_dict", None) is not None:
            self.ema.model.load_state_dict(
                self._pending_ema_state_dict
            )
            print("EMA restored from checkpoint")
            del self._pending_ema_state_dict

        print("EMA initialized:", self.ema is not None)
        print("EMA device:", next(self.ema.model.parameters()).device)

    def process_batch(self, batch, metrics):
        batch = super().process_batch(batch, metrics)

        if self.is_train:
            self.ema.update(self.model)

        return batch

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        super()._resume_checkpoint(resume_path)

        checkpoint = torch.load(
            resume_path,
            map_location=self.device,
            weights_only=False,
        )

        self._pending_ema_state_dict = checkpoint.get("ema_state_dict")

    def _train_epoch(self, epoch):
        logs = super()._train_epoch(epoch)

        original_model = self.model

        try:
            self.model = self.ema.model

            for part, dataloader in self.evaluation_dataloaders.items():
                ema_logs = super()._evaluation_epoch(
                    epoch, f"{part}_ema", dataloader
                )
                logs.update({
                    f"{part}_ema_{name}": value
                    for name, value in ema_logs.items()
                })

        finally:
            self.model = original_model

        return logs