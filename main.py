import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
#matplotlib inline

def main():
    labeled_images = pd.read_csv('./data/all/train.csv')
    images = labeled_images.iloc[0:5000,1:]
    labels = labeled_images.iloc[0:5000,:1]
    train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

    test_images[test_images>0]=1
    train_images[train_images>0]=1

    clf = svm.SVC()
    print(clf.fit(train_images, train_labels.values.ravel()))
    print(clf.score(test_images,test_labels))

if __name__ == '__main__':
    main()
