import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from mnist_dataset import MnistTest, MnistTrain

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on: {device}')


# Import data and choose if you want to use PCA
train_data = MnistTrain(pca=False)
test_data = MnistTest(pca=False)


# Initialize variables
learning_rate = 0.01
batch = 1000
n_epochs = 50
hidden_layers = 3
hidden_size = [64, 64, 64]


# Define the NN
class MLP(nn.Module):
    # Init function
    def __init__(self, hidden_layers=1, hidden_size=[254], activation='relu'):
        super(MLP, self).__init__()

        self.depth = nn.Sequential()

        # Make a dict for the activation functions
        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid()
        })

        # The input on the first layer is the number of columns of the dataset
        input_size = train_data.x.size(dim=1)

        # Iterate for the appropriate number of layers
        for i in range(hidden_layers):
            self.depth.add_module(
                f'hidden{i+1}', nn.Linear(input_size, hidden_size[i]))

            # Pass the activation function according to the activation variable
            self.depth.add_module(f'activation', self.activations[activation])

            # Change the input of the next layer
            input_size = hidden_size[i]

        # Add the last layer with 10 outputs
        self.depth.add_module(f'classifier', nn.Linear(
            hidden_size[hidden_layers-1], 10))

    # Forward function
    def forward(self, x):
        out = self.depth(x)
        return out


# Create the model and pass it to the device
model = MLP(hidden_layers=hidden_layers, hidden_size=hidden_size,
            activation='sigmoid').to(device)

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
