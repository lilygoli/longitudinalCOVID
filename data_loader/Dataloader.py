from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate
import numpy as np

class Dataloader(DataLoader):
    """
    data loading -- uncomment the commented lines for reverse weight sampling the classes
    """

    def __init__(self, dataset, batch_size, shuffle=True, num_workers=1):
        self.dataset = dataset
        # self.weights = np.array(self.dataset.number_of_classes)
        # self.weights = 1 / self.weights
        # self.weights = self.weights / sum(self.weights)
        # self.balance = self.dataset.weights

        self.shuffle = shuffle

        self.batch_idx = 0

        if self.shuffle:
            self.sampler = RandomSampler(self.dataset) # Replace with: WeightedRandomSampler(self.balance, len(self.dataset))
        else:
            self.sampler = SequentialSampler(self.dataset)
        self.shuffle = False

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': default_collate,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)
