import numpy as np


def make_one_hot(x, num_class=None):
    if not num_class:
        num_class = np.max(x) + 1
    ohx = np.zeros((len(x), num_class))
    ohx[range(len(x)), x] = 1
    return ohx


class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        dx = dout * (self.x > 0)
        return dx


class Sigmoid:
    def forward(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))

    def backward(self, dout):
        dx = dout * self.x * (1 - self.x)
        return dx


class Tanh:
    def forward(self, x):
        self.x = x
        return np.tanh(x)

    def backward(self, dout):
        dx = dout * (1 - self.x ** 2)
        return dx

class Swish:
    def forward(self, x):
        self.x = x
        return x * self.sigmoid(x)

    def backward(self, dout):
        return dout * self.sigmoid(self.x) * (1 - self.sigmoid(self.x))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.random([in_features, out_features]) * 0.001
        self.bias = np.zeros([1, out_features])
        self.out = None
        self.delta = None
        self.delta_weight = None
        self.delta_bias = None

    def forward(self, x):
        self.x = x
        self.out = np.dot(x, self.weight) + self.bias
        return self.out

    def backward(self, delta):
        self.delta = delta
        self.delta_weight = np.dot(self.x.T, self.delta)
        self.delta_bias = np.sum(self.delta, axis=0, keepdims=True)
        return np.dot(self.delta, self.weight.T)

    def update(self, lr):
        self.weight -= lr * self.delta_weight
        self.bias -= lr * self.delta_bias

class Softmax:
    def forward(self, x):
        self.x = x
        self.exp_x = np.exp(x)
        self.out = self.exp_x / np.sum(self.exp_x, axis=1, keepdims=True)
        return self.out

    def backward(self, delta):
        self.delta = delta
        self.delta_x = self.out * (1 - self.out) * self.delta
        return self.delta_x

class BatchNormalization:
    def __init__(self,
                 gamma,
                 beta,
                 momentum=0.9,
                 running_mean=None,
                 running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None
        self.running_mean = running_mean
        self.running_var = running_var
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, is_training=True):  #封装：修改卷积模式到FC模式
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, is_training)

        return out.reshape(*self.input_shape)  # *表示按照元组的形式返回

    def __forward(self, x, is_training):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if is_training:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (
                1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (
                1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((self.running_var + 10e-7)**0.5)

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

    def update(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta