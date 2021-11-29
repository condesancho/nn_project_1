""" This files contain two custom torch Datasets and loads the MNIST data from the samples folder """

import torch
from torch.utils.data import Dataset

import numpy as np

from mnist import MNIST

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class MnistTrain(Dataset):

    def __init__(self, pca=False, num_components=331):
        # Change './samples' to the path of the files
        mndata = MNIST('./samples')
        self.x, self.y = mndata.load_training()

        # Convert from list to numpy array
        self.x = np.array(self.x)

        # If you want to use PCA
        if pca:
            scaler = StandardScaler().fit_transform(self.x)
            # clf.fit(scaler)
            clf = PCA(num_components)
            self.x = clf.fit_transform(scaler)

        # Shape of x is [60000,784]
        self.x = torch.from_numpy(self.x).type(torch.FloatTensor)

        # Shape of y is [60000,]
        self.y = torch.from_numpy(np.array(self.y))
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class MnistTest(Dataset):

    def __init__(self, pca=False, num_components=331):
        # Change './samples' to the path of the files
        mndata = MNIST('./samples')
        self.x, self.y = mndata.load_testing()

        # Convert from list to numpy array
        self.x = np.array(self.x)

        if pca:
            scaler = StandardScaler().fit_transform(self.x)
            clf = PCA(num_components)
            self.x = clf.fit_transform(scaler)

        # Shape of x is [10000,784]
        self.x = torch.from_numpy(self.x).type(torch.FloatTensor)

        # Shape of y is [10000,]
        self.y = torch.from_numpy(np.array(self.y))
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
