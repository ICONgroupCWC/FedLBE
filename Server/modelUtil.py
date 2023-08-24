import torch
import torch.nn as nn
import numpy as np
SCALE_QUANTIZE = 0.1
ZERO_POINT_QUANTIZE = 0.0

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

# Function to dequantize a tensor
def dequantize_tensor(q_x, scale_quantize, zero_point_quantize, info=None):
    if info is None:
        return scale_quantize * (q_x.float() - zero_point_quantize)
    else:
        return info[0] * (q_x.float() - info[1]) + info[2]

def decompress_tensor(v, i, s):
    s = s.tolist()
    x_d = torch.zeros(np.prod(s)).to(v.device)
    x_d[i.long()] = v
    x_d = x_d.reshape(s)
    return x_d
