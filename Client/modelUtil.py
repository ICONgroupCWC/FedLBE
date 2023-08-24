import torch
import torch.nn as nn
import numpy as np
#
# SCALE_QUANTIZE = 0.1
# ZERO_POINT_QUANTIZE = 0.0


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


# Function to quantize a tensor
def quantize_tensor(x, scale_quantize, zero_point_quantize, num_bits=8, adaptive=False, info=None):
    qmin = -2. ** (num_bits - 1)
    qmax = 2. ** (num_bits - 1) - 1.
    if adaptive:
        min_val, max_val, mean_val = x.min(), x.max(), x.mean()

        scale = (max_val - min_val) / (qmax - qmin)
        if scale == 0.0:
            scale = 0.001

        initial_zero_point = qmin - (min_val - mean_val) / scale

        zero_point = 0
        if initial_zero_point < qmin:
            zero_point = qmin
        elif initial_zero_point > qmax:
            zero_point = qmax
        else:
            zero_point = initial_zero_point
        zero_point = int(zero_point)
    else:
        if info is not None:
            scale = info[0]
            zero_point = info[1]
            mean_val = info[2]
        else:
            scale = scale_quantize
            zero_point = zero_point_quantize
            mean_val = 0.0

    q_x = zero_point + (x - mean_val) / scale
    q_x.clamp_(qmin, qmax).round_()
    if num_bits == 8:
        q_x = q_x.round().char()
    elif num_bits == 16:
        q_x = q_x.round().short()
    return q_x, torch.tensor([scale, zero_point, mean_val])


def compress_tensor(x, r, comp_type):

    s = torch.tensor(x.shape).type(torch.int16)
    x_f = x.flatten()
    k = int(len(x_f) * r / 2)
    if k == 0:
        raise ValueError("Compression ratio is too low!")
    if comp_type == 'topk':
        v, i = x_f.abs().topk(k)
    elif comp_type == 'random':
        i = torch.randperm(len(x_f))[:k]
    v = x_f[i]
    i = i.type(torch.int32)
    return v, i, s


