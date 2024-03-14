import torch
from torchvision import transforms


def get_transformations(trainsforms):

    '''Providing custom transformations'''
    transformations = [transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32)]

    if 'normalize' in trainsforms and trainsforms['normalize']:
        print('normalizing')
        transformations.append(transforms.Normalize((float(trainsforms['mean']),), (float(trainsforms['std']),)))

    return transformations



