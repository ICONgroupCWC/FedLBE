from torch.utils.data import Dataset

from .TextDataset import TextDataset
from .ImageDataset import ImageDataset
from .LinearDataset import  LinearDataset
def getDataloader(dataset, labels, dataops):

    if dataops['dtype'] == 'img':
        return ImageDataset(dataset, labels, dataops)
    elif dataops['dtype'] == 'text':
        return TextDataset(dataset, labels)
    elif dataops['dtype'] == 'One D':
        return LinearDataset(dataset, labels)

