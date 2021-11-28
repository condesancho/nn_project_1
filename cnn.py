import torch
from torch import optim
from torch.autograd.grad_mode import F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataloader, dataset
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from mnist_dataset import MnistTest, MnistTrain

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import data
train_data = MnistTrain()
test_data = MnistTest()

# Data needs to be modified to 3D arrays
train_data.x = train_data.x.reshape(train_data.__len__(), 1, 28, 28)
test_data.x = test_data.x.reshape(test_data.__len__(), 1, 28, 28)

# Initialize variables
learning_rate = 0.01
batch = 1000
n_epochs = 25


# Define the NN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # The size of the image stays the same with 8 output channels
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)

        # The size of the image is divided by 2
        self.pool = nn.MaxPool2d(2)

        # The size of the image stays the same with 16 output channels
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)

        # The output of the last conv layer is 16 channels by 7x7 image
        self.final_layer1 = nn.Linear(16*7*7, 64)
        self.final_layer2 = nn.Linear(64, 10)

    def forward(self, x):
        # Pass through the first two conv layers
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))

        # Change the output of the conv layers to 2D tensor
        out = out.view(-1, 16*7*7)

        # Pass through the two fully connected layers
        out = F.relu(self.final_layer1(out))
        out = self.final_layer2(out)
        return out


# Create the model and pass it to the device
model = CNN().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Create dataloaders
train_loader = DataLoader(dataset=train_data, batch_size=batch,
                          shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_data, batch_size=batch,
                         shuffle=True, num_workers=2)


# Variables that store the training and the test losses
test_losses = []
train_losses = []

test_acc = []
train_acc = []


# Training function
def train(n_epochs):

    temp_loss = 0
    n_correct = 0
    n_samples = 0

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

        # Accumulate loss
        temp_loss += loss.item()

        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    # Store the mean loss of the epoch
    train_loss = temp_loss / len(train_loader)
    train_losses.append(train_loss)

    # Store the accuracy of the train set
    acc = 100.0 * n_correct / n_samples
    train_acc.append(acc)

    # Print the values
    print(f'Epoch: {epoch+1}, Loss: {train_loss:.4f} and Acc: {acc:.2f}')


# Testing function
def test():
    with torch.no_grad():
        temp_loss = 0
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            # Accumulate loss
            temp_loss += loss.item()

            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

        # Store the mean loss of the epoch
        test_loss = temp_loss / len(train_loader)
        test_losses.append(test_loss)

        # Store the accuracy of the train set
        acc = 100.0 * n_correct / n_samples
        test_acc.append(acc)

        # Print the values
        print(f'Test Loss: {test_loss:.4f} and Test Acc: {acc:.2f}\n')


# Iterate for every epoch
for epoch in range(n_epochs):
    train(epoch)
    test()


# Plot accuracies
plot1 = plt.figure(1)
plt.plot(train_acc, '-o')
plt.plot(test_acc, '-o')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Train', 'Test'])
plt.title('Train vs Test Accuracy')


# Plot losses
plot2 = plt.figure(2)
plt.plot(train_losses, '-o')
plt.plot(test_losses, '-o')
plt.xlabel('epoch')
plt.ylabel('losses')
plt.legend(['Train', 'Test'])
plt.title('Train vs Test Losses')

plt.show()
