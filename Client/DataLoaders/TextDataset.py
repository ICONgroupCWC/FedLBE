import torch
from torch.utils.data import Dataset
import numpy as np


class TextDataset(Dataset):

    def __init__(self, dataset, labels):
        super().__init__()
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        indices, target = self.dataset[index], self.labels[index]
        # indices = word_to_indices(sentence)
        # target = letter_to_vec(target)

        # target = [_one_hot(i) for i in target]

        target = torch.as_tensor(target).long()

        target = torch.nn.functional.one_hot(target, num_classes=80)

        # print('target -' + str(target[0]))
        # print(target)
        indices = torch.LongTensor(np.array(indices))
        # target = torch.nn.functional.one_hot(torch.as_tensor(target), num_classes=80)

        return indices, torch.ravel(target).float()
