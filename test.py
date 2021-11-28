import torch
from torch._C import device
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, dataloader, dataset

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from mnist_dataset import MnistTest, MnistTrain

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import data
train_data = MnistTrain()
test_data = MnistTest()

# Create dataloaders
train_loader = DataLoader(dataset=train_data, batch_size=10,
                          shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_data, batch_size=10,
                         shuffle=True, num_workers=2)

std_x = StandardScaler().fit_transform(train_data.x)
pca = PCA(n_components=100)
principal_components = pca.fit_transform(std_x)

print(principal_components.shape)

train_data.x = torch.from_numpy(principal_components).type(torch.FloatTensor)
print(type(train_data.x))
print(train_data.x.type())
