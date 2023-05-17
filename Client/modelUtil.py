import torch
import torch.nn as nn

def get_optimizer(op_type, model, lr):

    if op_type == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    elif op_type == 'Adadelta':
        return torch.optim.Adadelta(model.parameters(), lr=lr)
    elif op_type == 'Adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    elif op_type == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif op_type == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif op_type == 'SparseAdam':
        return torch.optim.SparseAdam(model.parameters(), lr=lr)
    elif op_type == 'Adamax':
        return torch.optim.Adamax(model.parameters(), lr=lr)
    elif op_type == 'ASGD':
        return torch.optim.ASGD(model.parameters(), lr=lr)
    elif op_type == 'LBFGS':
        return torch.optim.LBFGS(model.parameters(), lr=lr)
    elif op_type == 'NAdam':
        return torch.optim.NAdam(model.parameters(), lr=lr)
    elif op_type == 'RAdam':
        return torch.optim.RAdam(model.parameters(), lr=lr)
    elif op_type == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    elif op_type == 'Rprop':
        return torch.optim.Rprop(model.parameters(), lr=lr)

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
