import sys

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from basic import BatchNormalization, Linear, ReLU, Sigmoid, Tanh,Swish, make_one_hot,Softmax

sys.path.append('mnist')
import mnist


# hiden layer == 3
class MLP:
    def __init__(self,
                 lr,
                 grad='relu',
                 optimizer='SGD',
                 in_features=784,
                 out_features=10):
        #输入的28*28为图片大小，输出的10为数字的类别数
        hiddendims1 = 512
        hiddendims2 = 256
        hiddendims3 = 256
        if grad == 'relu':
            self.AF1 = ReLU()
            self.AF2 = ReLU()
            self.AF3 = ReLU()
        elif grad == 'sigmoid':
            self.AF1 = Sigmoid()
            self.AF2 = Sigmoid()
            self.AF3 = Sigmoid()
        elif grad == 'tanh':
            self.AF1 = Tanh()
            self.AF2 = Tanh()
            self.AF3 = Tanh()
        elif grad == 'swish':
            self.AF1 = Swish()
            self.AF2 = Swish()
            self.AF3 = Swish()
        self.linear1 = Linear(in_features, hiddendims1)
        self.BN1 = BatchNormalization(gamma=1, beta=0)
        self.linear2 = Linear(hiddendims1, hiddendims2)
        self.BN2 = BatchNormalization(gamma=1, beta=0)
        self.linear3 = Linear(hiddendims2, hiddendims3)
        self.BN3 = BatchNormalization(gamma=1, beta=0)
        self.linear4 = Linear(hiddendims3, out_features)
        self.softmax = Softmax()
        self.lr = lr

    def forward(self, x, is_training=True):
        x = self.linear1.forward(x)
        x = self.BN1.forward(x, is_training)
        x = self.AF1.forward(x)
        x = self.linear2.forward(x)
        x = self.BN2.forward(x, is_training)
        x = self.AF2.forward(x)
        x = self.linear3.forward(x)
        x = self.BN3.forward(x, is_training)
        x = self.AF3.forward(x)
        x = self.linear4.forward(x)
        x=  self.softmax.forward(x)
        return x

    def backward(self, x, y):
        out = self.forward(x)
        y = make_one_hot(y)
        dx = (out - y) / x.shape[0]
        dx = self.softmax.backward(dx)
        dx = self.linear4.backward(dx)
        dx = self.AF3.backward(dx)
        dx = self.BN3.backward(dx)
        dx = self.linear3.backward(dx)
        dx = self.AF2.backward(dx)
        dx = self.BN2.backward(dx)
        dx = self.linear2.backward(dx)
        dx = self.AF1.backward(dx)
        dx = self.BN1.backward(dx)
        dx = self.linear1.backward(dx)
        return dx
        
    def update(self):
        self.linear1.update(self.lr)
        self.BN1.update(self.lr)
        self.linear2.update(self.lr)
        self.BN2.update(self.lr)
        self.linear3.update(self.lr)
        self.BN3.update(self.lr)
        self.linear4.update(self.lr)

    def train(self, x, y, epochs=10, batch_size=128):
        lx = []
        loss=[]
        accuracy=[]
        for epoch in range(epochs):
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                dx = self.backward(x_batch, y_batch)
                self.update()
            lx.append(epoch)
            loss.append(self.loss(x, y))
            accuracy.append(self.accuracy(x, y))
            print('epoch:', epoch+1, 'loss:', self.loss(x, y), 'accuracy:', self.accuracy(x, y))
        # draw loss and accuracy
        plt.figure(1)
        plt.plot(lx, loss, 'r-')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('train loss')
        plt.savefig("train_loss.png")
        plt.figure(2)
        plt.plot(lx, accuracy, 'b-')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('train accuracy')
        plt.savefig("train_accuarcy.png")


    
    def predict(self, x):
        out = self.forward(x, is_training=False)
        return np.argmax(out, axis=1)

    def accuracy(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred == y)

    def loss(self, x, y):
        out = self.forward(x,is_training=False)
        y = make_one_hot(y)
        loss = np.mean(np.square(out - y))
        return loss


if __name__ == '__main__':
    epochs = 10
    lr = 0.1
    batch_size = 128
    net = MLP(lr, grad='sigmoid')

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    n_train, w, h = train_images.shape
    X_train = train_images.reshape((n_train, w * h))
    Y_train = train_labels

    n_test, w, h = test_images.shape
    X_test = test_images.reshape((n_test, w * h))
    Y_test = test_labels

    net.train(X_train, Y_train, epochs=epochs, batch_size=batch_size)
    print('eval accuracy:', net.accuracy(X_test, Y_test))
