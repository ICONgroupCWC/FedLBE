import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .preprocessUtil import get_transformations

class ImageDataset(Dataset):

    def __init__(self, dataset, labels, transformations):
        super().__init__()
        self.dataset = dataset
        self.labels = labels
        self.transform = transforms.Compose(get_transformations(transformations))
        self.target_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index]
        image = self.transform(image)
        label = torch.tensor(self.labels[index]).type(torch.LongTensor)
        return image, label