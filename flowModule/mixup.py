import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x).type(torch.long)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, mixed_y


def mixup_criterion(pred, y_a, y_b, lam = 0.5):
    criterion = nn.CrossEntropyLoss()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)