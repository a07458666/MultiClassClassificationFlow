import os

import numpy as np


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path, mode=0o777)


def position_encode(X, m=3, axis=1):
    x_p_list = [X]
    for i in range(m):
        x_p_list.append(np.sin((2**(i+1)) * X))
        x_p_list.append(np.cos((2**(i+1)) * X))
    return np.concatenate(x_p_list, axis=axis)
