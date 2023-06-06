from torch.utils.data import Dataset

from .TextDataset import TextDataset
from .ImageDataset import ImageDataset

def getDataloader(dataset, labels, dataops):

    if dataops['dtype'] == 'img':
        return ImageDataset(dataset, labels, dataops)
    elif dataops['dtype'] == 'text':
        return TextDataset(dataset, labels)

