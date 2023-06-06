import torch
from torchvision import transforms


def get_transformations(trainsforms):
    transformations = [transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32)]

    if trainsforms['normalize']:
        print('normalizing')
        transformations.append(transforms.Normalize((float(trainsforms['mean']),), (float(trainsforms['std']),)))

    return transformations



