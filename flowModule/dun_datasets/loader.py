from os import path
from torch.utils.data import Dataset
from numpy.random import uniform, randn
import numpy as np
import torch.utils.data as data

import zipfile
import pickle
try:
    import urllib
    from urllib import urlretrieve
except Exception:
    import urllib.request as urllib
from os import path

import numpy as np
from numpy.random import uniform, randn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from .additional_gap_loader import load_my_1d, load_wiggle_1d, load_agw_1d, load_matern_1d
from .additional_gap_loader import load_origin, load_axis

class MyDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

    def __len__(self):
        return len(self.x)

def loadDataset(datasetName):
    if(datasetName == "my_1d"):
        X_train, y_train, X_test, y_test = load_my_1d("./dataset")
    elif (datasetName == "wiggle"):
        X_train, y_train = load_wiggle_1d()
    elif (datasetName == "matern"):
        X_train, y_train = load_matern_1d("./dataset")
    elif (datasetName == "agw"):
        X_train, y_train = load_agw_1d("./dataset")
    elif (datasetName == "origin"):
        X_train, y_train = load_origin("./dataset")
    elif (datasetName == "axis"):
        X_train, y_train = load_axis("./dataset")  
    return X_train, y_train