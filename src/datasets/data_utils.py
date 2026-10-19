"""Build datasets, dataloaders, samplers, and batch transforms"""
from itertools import repeat

import torch
from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed
from src.datasets.sortagrad_sampler import SortaGradSampler



def inf_loop(dataloader):
    """
    Yield batches from a dataloader indefinitely

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def _get_seed(config):
    """
    Get reproducibility seed for training or inference
    """
    if config.get("trainer") is not None:
        return config.trainer.get("seed", 42)

    if config.get("inferencer") is not None:
        return config.inferencer.get("seed", 42)

    return 42

def get_dataloaders(config, text_encoder, device):
    """
    Create dataloaders for each of the dataset partitions
    Also creates instance and batch transforms

    Args:
        config (DictConfig): hydra experiment config.
        text_encoder (CTCTextEncoder): instance of the text encoder
            for the datasets.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataloaders init
    dataloaders = {}
    seed = _get_seed(config)
    for dataset_partition in config.datasets.keys():
        # dataset partition init
        dataset = instantiate(
            config.datasets[dataset_partition], text_encoder=text_encoder
        )  # instance transforms are defined inside


        if dataset_partition == "train":
            sampler = SortaGradSampler(dataset, seed=seed)
            shuffle = False
        else:
            sampler = None
            shuffle = False

        generator = torch.Generator()
        generator.manual_seed(
            seed + (0 if dataset_partition == "train" else 1)
        )
        
        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            sampler=sampler,
            shuffle=shuffle,
            worker_init_fn=set_worker_seed,
            generator=generator,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
