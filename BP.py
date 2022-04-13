import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

from basic import ReLU, Sigmoid, Tanh


class bpNN:
    def __init__(self,
                 input_dim,
                 numH,
                 numO,
                 learning_rate,
                 reg_lambda,
                 grad='relu'):
        self.numH = numH
        self.numO = numO
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        #初始化权重和偏置
        self.weight_H = np.random.random([input_dim, self.numH]) * 0.001
        self.b_H = np.zeros([1, self.numH])
        self.weight_O = np.random.random([self.numH, self.numO]) * 0.001
        self.b_O = np.zeros([1, self.numO])
        self.AF = None
        if grad == 'sigmoid':
            self.AF = Sigmoid()
        elif grad == 'tanh':
            self.AF = Tanh()
        elif grad == 'relu':
            self.AF = ReLU()

    def forward(self, x):
        z1 = np.dot(x, self.weight_H) + self.b_H
        a1 = self.AF.forward(z1)
        z2 = np.dot(a1, self.weight_O) + self.b_O
        z = z2 - np.max(z2)
        exp_scores = np.exp(z)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return z1, a1, z2, probs

    def train(self, x, y):  #训练函数, BP误差传递主体
        z1, a1, z2, probs = self.forward(x)
        delta3 = probs - y
        dw2 = np.dot(a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.weight_O.T) * self.AF.backward()
        dw1 = np.dot(x.T, delta2)
        db1 = np.sum(delta2, axis=0)
        # dw2 += self.reg_lambda * self.weight_O
        # dw1 += self.reg_lambda * self.weight_H
        self.weight_H -= self.learning_rate * dw1
        self.b_H -= self.learning_rate * db1
        self.weight_O -= self.learning_rate * dw2
        self.b_O -= self.learning_rate * db2

    def predict(self, x):  #预测函数
        z1, a1, z2, probs = self.forward(x)
        return np.argmax(probs, axis=1)

    def accuracy(self, x, y):
        predict_y = self.predict(x)
        acc = 1 - np.sum(abs(predict_y - y)) / len(y)
        return acc


def make_one_hot(x, num_class=None):
    if not num_class:
        num_class = np.max(x) + 1
    ohx = np.zeros((len(x), num_class))
    ohx[range(len(x)), x] = 1
    return ohx


def draw(X, y, name='halmoon.png'):
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.savefig(name)


def draw_p(X, y, net, name='halmoon_predict.png'):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.floor(net.predict(np.c_[xx.ravel(), yy.ravel()]) * 1.99999)
    Z = Z.reshape(xx.shape)
    #预测分界图像s
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.savefig(name)


if __name__ == '__main__':
    num = 10000
    batch_size = 64
    epoch = 500
    train_num = int(num * 0.8)
    test_num = num - train_num
    X, y = datasets.make_moons(num, noise=0.2)  #产生数据集
    draw(X, y)  #画数据集
    train_X, train_y = np.array(X[:train_num]), np.array(y[:train_num])
    test_X, test_y = np.array(X[train_num:]), np.array(y[train_num:])

    net = bpNN(2, 10, 2, 0.01, 0.01, grad='relu')
    for i in range(epoch):
        for j in range(0, train_num, batch_size):
            net.train(train_X[j:j + batch_size],
                      make_one_hot(train_y[j:j + batch_size]))
        print('epoch:', i, 'acc:', net.accuracy(test_X, test_y))
    draw_p(test_X, test_y, net, 'halmoon_predict.png')
