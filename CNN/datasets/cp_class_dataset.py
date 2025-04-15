"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it), 
    Davide Zoni (davide.zoni@polimi.it)
"""

import os
import numpy as np
from torch.utils.data import Dataset


class CpClassifierDataset(Dataset):
    def __init__(self, data_dir, labels,
                 which_subset='train'):

        self.labels = labels
        print("Loading {} set..".format(which_subset) + " in " +
              data_dir + '/' + '{}_set.npy'.format(which_subset))

        self.windows = np.load(
            os.path.join(data_dir, '{}_set.npy'.format(which_subset)), mmap_mode='r+')

        self.target = np.load(
            os.path.join(data_dir, '{}_labels.npy'.format(which_subset)),  mmap_mode='r+')

        print("Number of windows in the {} set: {}".format(
            which_subset, self.windows.shape[0]))

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, index):
        x = (self.windows[index] - np.mean(self.windows[index])
             ) / np.std(self.windows[index])

        which_target = self.target[index]
        y = int(self.labels[which_target])

        return x, y