from torch.utils.data import Dataset
from config import *
from utils import *

class Dataset(Dataset):
    def __init__(self, dataset=None, mode=None, label_type=None):
        super(Dataset, self).__init__()
        
        assert dataset in ['local', 'open'], 'type error'
        assert mode in ['train', 'val', 'test'], 'mode error'
        assert label_type in ['digit', 'distance'], 'label error'

        print(f'build {dataset} samples ({mode})...')

        self.path = build_path(dataset, mode)
        self.samples = build_samples(self.path, label_type)

        print(f'count samples ({mode}) : {len(self.samples)}')

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
