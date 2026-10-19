"""Train and evaluate an ASR model from a Hydra configuration"""
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.trainer.ema_trainer import EMATrainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Build the ASR training pipeline and start model training

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    requested_device = config.trainer.get("device", "auto")

    if requested_device != "auto":
        device = str(requested_device)
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Device:", device)

    # setup text_encoder
    text_encoder = instantiate(config.text_encoder)
    if "lm" in config and config.lm.get("enabled", False):
        print(f"Loading language model from: {config.lm.path}")

        text_encoder.load_lm(
            config.lm.path,
            alpha=config.lm.alpha,
            beta=config.lm.beta,
        )

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

    # build model architecture, then print to console
    model = instantiate(config.model, n_tokens=len(text_encoder)).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)

    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            # use text_encoder in metrics
            metrics[metric_type].append(
                instantiate(metric_config, text_encoder=text_encoder)
            )

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)

    epoch_len = config.trainer.get("epoch_len")

    if epoch_len is None:
        steps_per_epoch = len(dataloaders["train"])
    else:
        steps_per_epoch = epoch_len

    lr_scheduler = instantiate(
        config.lr_scheduler,
        optimizer=optimizer,
        steps_per_epoch=steps_per_epoch,
    )

    trainer_cls = (
        EMATrainer
        if config.trainer.get("use_ema", False)
        else Trainer
    )
    
    trainer = trainer_cls(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        text_encoder=text_encoder,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )


    trainer.train()


if __name__ == "__main__":
    main()
