import pandas as pd
#import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

import sys
sys.path.append('./data/')
import model as m
import torchvision
import torchvision.transforms as transforms

#matplotlib inline

def main():
    labeled_images = pd.read_csv('./data/all/train.csv')
    images = labeled_images.iloc[0:5000,1:]
    labels = labeled_images.iloc[0:5000,:1]
    train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

    test_images[test_images>0]=1
    train_images[train_images>0]=1

    """
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
                                           batch_size=batch_size,
                                           shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

    model = m.Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters())

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

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



if __name__ == '__main__':
    main()
