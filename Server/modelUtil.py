import torch
import torch.nn as nn



def get_criterion(criterion):

    if criterion == 'L1Loss':
        return nn.L1Loss()
    elif criterion == 'MSELoss':
        return nn.MSELoss()
    elif criterion == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif criterion == 'BCELoss':
        return nn.BCELoss()
    elif criterion == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
