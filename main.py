import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

import sys
sys.path.append('./data/')
import torch
from torch import autograd, nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

#matplotlib inline

def main():
    """
    labeled_images = pd.read_csv('./data/all/train.csv')
    images = labeled_images.iloc[0:5000,1:]
    labels = labeled_images.iloc[0:5000,:1]
    train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

    test_images[test_images>0]=1
    train_images[train_images>0]=1


    # SVM
    clf = svm.SVC()
    print(clf.fit(train_images, train_labels.values.ravel()))
    print(clf.score(test_images,test_labels))
    """

    # CNN

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=8,
                                           shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=8,
                                          shuffle=False)

    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters())

    # Train the model
    total_step = len(train_loader)
    num_epochs = 2
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            #images = images.to(device)
            #labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

class Net(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=4, kernel_size=5, padding=0)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.conv2 = nn.Conv2d(in_channels=4,out_channels=8, kernel_size=5, padding=0)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.lin = nn.Linear(in_features=8*batch_size, out_features=10)

    def forward(self, inp):
        y = self.conv1(inp)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool(y)
        y = y.reshape(y.size(0), -1)
        print('shape(y)', np.shape(y))
        return self.lin(y)

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    main()
