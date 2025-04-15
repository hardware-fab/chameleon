"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it), 
    Davide Zoni (davide.zoni@polimi.it)
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from CNN.datasets import CpClassifierDataset


class CpClassifierDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, labels):
        super(CpClassifierDataModule, self).__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.labels = labels

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_set = CpClassifierDataset(
                self.data_dir, self.labels
            )
            self.valid_set = CpClassifierDataset(
                self.data_dir, self.labels, which_subset='valid'
            )
        elif stage == 'test':
            self.test_set = CpClassifierDataset(
                self.data_dir, self.labels, which_subset='test'
            )
        else:
            raise Exception('Unsupported stage type')

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        return [DataLoader(self.valid_set,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           shuffle=False),]

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)
