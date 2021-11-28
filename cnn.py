import torch
from torch import optim
from torch.autograd.grad_mode import F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataloader, dataset
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from mnist_dataset import MnistTest, MnistTrain

import sys

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import data
train_data = MnistTrain()
test_data = MnistTest()

# Data needs to be modified to 3D arrays
train_data.x = train_data.x.reshape(train_data.__len__(), 1, 28, 28)
test_data.x = test_data.x.reshape(test_data.__len__(), 1, 28, 28)

# Initialize variables
learning_rate = 0.001
batch = 1000
n_epochs = 10


# Define the NN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)

        self.final_layer1 = nn.Linear(16*7*7, 64)
        self.final_layer2 = nn.Linear(64, 10)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = out.view(-1, 16*7*7)
        out = F.relu(self.final_layer1(out))
        out = self.final_layer2(out)
        return out


# Create the model and pass it to the device
model = CNN().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create dataloaders
train_loader = DataLoader(dataset=train_data, batch_size=batch,
                          shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_data, batch_size=batch,
                         shuffle=True, num_workers=2)

# Training
total_steps = len(train_loader)
for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Pass data to the device
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 60 == 0:
            print(
                f'epoch {epoch+1} / {n_epochs}, step {i+1}/{total_steps}, loss = {loss.item():.4f}')

# Testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')
