import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
class CustomDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.ConvertImageDtype(torch.float32 ) ,  transforms.Normalize((0.1307,), (0.3081,))])
        self.target_transform = transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # print('getting index ' + str(index))
        image = self.dataset[index]
        image = self.transform(image)
        # print('image type ' + str(type(image)))
        label = torch.tensor(self.labels[index]).type(torch.LongTensor)
        return image, label