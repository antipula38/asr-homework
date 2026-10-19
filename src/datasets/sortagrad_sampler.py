import torch

class SortaGradSampler:
    """
    Use length-sorted samples first, then random ordering
    """
    def __init__(self, dataset, seed):
        """
        Initialize the sampler for a dataset and random seed
        """
        self.epoch = 0
        self.dataset = dataset
        self.seed = seed

    def set_epoch(self, epoch):
        """
        Set the current training epoch
        """
        self.epoch = epoch

    def __iter__(self):
        """
        Return indices in sorted or deterministic random order
        """
        if self.epoch == 1:
            return iter(range(self.__len__()))
        else:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            return iter(torch.randperm(self.__len__(), generator=generator).tolist())

    def __len__(self):
        """
        Return the number of dataset samples
        """
        return len(self.dataset)