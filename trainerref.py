import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt


# Load the data:
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Split the data into features and labels for each image:
Y_train = train["label"]
X_train = train.drop(labels="label",axis=1)
X_test = test

# Normalize values:
X_train = X_train/255.0
test = test/255.0

# Reshape the pixels to their original 28x28 format:
X_train, Y_train = X_train.values.reshape(-1, 28,28), Y_train.values
X_test = X_test.values.reshape(-1,28,28)

# Split into training and validation:
X_valid, Y_valid = X_train[-2000:], Y_train[-2000:]
X_train, Y_train = X_train[:-2000], Y_train[:-2000]

trainset = MNISTDataset(X_train, Y_train)
testset = MNISTDataset(X_test)
validset = MNISTDataset(X_valid, Y_valid)

# batch_size=how many samples per batch to load
# num_workers= how many subprocesses to use for data loading
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
validloader = DataLoader(validset, batch_size=4, shuffle=False)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

for i, data in enumerate(trainloader):
    imgs, labels = data
    plt.figure()
    two_d = (np.reshape(imgs[0].numpy()*255, (28, 28))).astype(np.uint8)
    plt.imshow(two_d, cmap='gray')
    plt.show()
    print("Label: ", classes[labels[0]])
    if i >= 3:
        break

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5) # stride=1, number of filters=10, filter size=5
        self.pool = nn.MaxPool2d(2,2) # stride=2, filter size=2
        self.conv2 = nn.Conv2d(10, 20, 5) #stride=1, number of filters=20, filter size=5
        self.fc1 = nn.Linear(20*4*4, 50) # fully connected layer with input size 4x4x20 and output size 50
        self.fc2 = nn.Linear(50, 10) # fully connected layer with input size 50 and output size 10

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.float()
        a1 = self.pool(F.relu(self.conv1(x)))
        a2 = self.pool(F.relu(self.conv2(a1)))
        a3 = a2.view(-1, 20*4*4)
        a4 = F.relu(self.fc1(a3))
        a5 = self.fc2(a4)
        return a5

cnn = CNN()

learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.5)

def train(model, optimizer=optimizer, criterion=criterion, learning_rate=learning_rate, epochs=5):
    losses = []
    for epoch in range(epochs):
        for i, data in enumerate(trainloader):
            samples, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            predictions = model(samples)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            # Print some statistics
            if i % 2000 == 1999:
                losses.append(loss.data.mean())
                print('epoch[%d], mini-batch[%5d] loss: %.3f' % (epoch+1, i+1, np.mean(losses)))
                total_loss = 0

train(cnn)

total = 0
corrects = 0

for data in validloader:
    images, labels = data
    outputs = cnn(images)
    _, predicted_lables = torch.max(outputs.data, 1)
    total += labels.size(0)
    corrects += (predicted_lables == labels).sum().item()

print('Accuracy on %d test images is = %d %%' % (total, 100*corrects/total))

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

for i, data in enumerate(validloader):
    imgs, labels = data
    plt.figure()
    two_d = (np.reshape(imgs[0].numpy()*255, (28, 28))).astype(np.uint8)
    plt.imshow(two_d, cmap='gray')
    plt.show()
    outputs = cnn(imgs)
    _, predicted_label = torch.topk(outputs[0], 1)
    print('Predicted Label:', predicted_label.numpy())
    if i >= 3:
        break
