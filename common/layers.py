import numpy as np
from common.util import im2col, col2im

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        x[self.mask] = 0
        return x

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class MaxPool2D:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        PH, PW = self.pool_size, self.pool_size
        H_out = (H - PH) // self.stride + 1
        W_out = (W - PW) // self.stride + 1

        out = np.zeros((N, C, H_out, W_out))
        self.cache = (x, H_out, W_out)
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + PH
                w_start = j * self.stride
                w_end = w_start + PW
                pool_region = x[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.max(pool_region, axis=(2, 3))
        return out

    def backward(self, dout):
        x, H_out, W_out = self.cache
        N, C, H, W = x.shape
        PH, PW = self.pool_size, self.pool_size
        dx = np.zeros_like(x)

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + PH
                w_start = j * self.stride
                w_end = w_start + PW
                pool_region = x[:, :, h_start:h_end, w_start:w_end]
                max_vals = np.max(pool_region, axis=(2, 3), keepdims=True)
                mask = (pool_region == max_vals)
                dx[:, :, h_start:h_end, w_start:w_end] += mask * dout[:, :, i:i + 1, j:j + 1]
        return dx


class Flatten:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.cache)


class Dense:
    def __init__(self, in_features, out_features, name=None):
        self.name = name
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.b = np.zeros(out_features)
        self.dW = None
        self.db = None
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        x = self.cache
        self.dW = np.dot(x.T, dout)
        self.db = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, name=None):
        self.name = name
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2. / (in_channels * kernel_size**2))
        self.b = np.zeros(out_channels)
        self.stride = stride
        self.padding = padding
        self.dW = None
        self.db = None
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        KH, KW = self.W.shape[2], self.W.shape[3]
        self.cache = x
        x_col = im2col(x, KH, self.stride, self.padding)
        W_col = self.W.reshape(self.W.shape[0], -1).T
        out = x_col.dot(W_col) + self.b
        H_out = (H + 2 * self.padding - KH) // self.stride + 1
        W_out = (W + 2 * self.padding - KW) // self.stride + 1
        out = out.reshape(N, H_out, W_out, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        x = self.cache
        KH, KW = self.W.shape[2], self.W.shape[3]
        N, C, H, W = x.shape
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, self.W.shape[0])
        x_col = im2col(x, KH, self.stride, self.padding)
        self.dW = (x_col.T.dot(dout_reshaped)).T.reshape(self.W.shape)
        self.db = np.sum(dout_reshaped, axis=0)
        dx_col = dout_reshaped.dot(self.W.reshape(self.W.shape[0], -1))
        dx = col2im(dx_col, x.shape, KH, self.stride, self.padding)
        return dx

class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.9, name=None):
        self.name = name
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.cache = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, is_training=True):
        if is_training:
            mu = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)
            x_norm = (x - mu) / np.sqrt(var + self.eps)
            out = self.gamma.reshape(1, -1, 1, 1) * x_norm + self.beta.reshape(1, -1, 1, 1)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.squeeze()
            self.cache = (x, x_norm, mu, var)
        else:
            x_norm = (x - self.running_mean.reshape(1, -1, 1, 1)) / np.sqrt(self.running_var.reshape(1, -1, 1, 1) + self.eps)
            out = self.gamma.reshape(1, -1, 1, 1) * x_norm + self.beta.reshape(1, -1, 1, 1)
        return out

    def backward(self, dout):
        x, x_norm, mu, var = self.cache
        N, C, H, W = x.shape
        self.dgamma = np.sum(dout * x_norm, axis=(0, 2, 3))
        self.dbeta = np.sum(dout, axis=(0, 2, 3))
        dx_norm = dout * self.gamma.reshape(1, -1, 1, 1)
        dvar = np.sum(dx_norm * (x - mu) * (-0.5) * (var + self.eps) ** (-1.5), axis=(0, 2, 3), keepdims=True)
        dmu = np.sum(dx_norm * (-1 / np.sqrt(var + self.eps)), axis=(0, 2, 3), keepdims=True) + dvar * np.mean(
            -2 * (x - mu), axis=(0, 2, 3), keepdims=True)
        dx = dx_norm / np.sqrt(var + self.eps) + dvar * 2 * (x - mu) / (N * H * W) + dmu / (N * H * W)
        return dx

class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x, is_training=True):
        if is_training:
            self.mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * self.mask / (1 - self.dropout_rate)
        return x

    def backward(self, dout):
        return dout * self.mask / (1 - self.dropout_rate)