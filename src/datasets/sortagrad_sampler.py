import torch

class SortaGradSampler:
    def __init__(self, dataset, seed):
        self.epoch = 0
        self.dataset = dataset
        self.seed = seed

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.epoch == 1:
            return iter(range(self.__len__()))
        else:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            return iter(torch.randperm(self.__len__(), generator=generator).tolist())

    def __len__(self):
        return len(self.dataset)