import numpy as np
from torch.utils.data import Dataset

class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)

        self.dim_agg = self.datasets[0].dim_agg
        self.dim_point = self.datasets[0].dim_point
        self.dim_render = self.datasets[0].dim_render

    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                data = self.datasets[i][index]
                data['dataset_i'] = np.ones_like(data['object_id']) * i # from 0, 1, 2 ...
                return data
        raise IndexError(f'{index} exceeds {self.length}')

    def __len__(self):
        return self.length

    def setNormalization(self, *args, **kwargs):
        for dataset in self.datasets:
            dataset.setNormalization(*args, **kwargs)
