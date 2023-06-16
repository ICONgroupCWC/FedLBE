from .TextDataset import TextDataset
from .ImageDataset import ImageDataset

def getDataloader(dataset, labels, dataops):

    '''Selecting data loader according to the data type'''

    if dataops['dtype'] == 'img':
        return ImageDataset(dataset, labels, dataops)
    elif dataops['dtype'] == 'text':
        return TextDataset(dataset, labels)

