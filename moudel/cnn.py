import numpy as np
from common.layers import *


class CNN:
    def __init__(self, use_dropout=True):
        self.layers = [
            Conv2D(3, 64, 3, stride=1, padding=1, name='conv1'),
            BatchNorm(64, name='bn1'),
            ReLU(),
            MaxPool2D(2, 2),
            Conv2D(64, 128, 3, stride=1, padding=1, name='conv2'),
            BatchNorm(128, name='bn2'),
            ReLU(),
            MaxPool2D(2, 2),
            Conv2D(128, 256, 3, stride=1, padding=1, name='conv3'),
            BatchNorm(256, name='bn3'),
            ReLU(),
            MaxPool2D(2, 2),
            Flatten(),
            Dense(256 * 4 * 4, 512, name='dense1'),
            ReLU(),
            Dropout(0.5) if use_dropout else None,
            Dense(512, 10, name='dense2')
        ]
        self.layers = [layer for layer in self.layers if layer is not None]
        self.params = {}
        for layer in self.layers:
            if isinstance(layer, (Conv2D, Dense, BatchNorm)):
                if hasattr(layer, 'W'):
                    self.params[f'{layer.name}_W'] = layer.W
                    self.params[f'{layer.name}_b'] = layer.b
                if isinstance(layer, BatchNorm):
                    self.params[f'{layer.name}_gamma'] = layer.gamma
                    self.params[f'{layer.name}_beta'] = layer.beta
        self.prob = None

    def predict(self, x, is_training=False):
        for layer in self.layers:
            if isinstance(layer, (BatchNorm, Dropout)):
                x = layer.forward(x, is_training=is_training)
            else:
                x = layer.forward(x)
        self.prob = self._softmax(x)
        return self.prob

    def _softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def loss(self, x, y_true):
        prob = self.predict(x, is_training=True)
        return -np.mean(y_true * np.log(prob + 1e-7))

    def accuracy(self, x, y_true):
        prob = self.predict(x, is_training=False)
        y_pred = np.argmax(prob, axis=1)
        y_true = np.argmax(y_true, axis=1)
        return np.mean(y_pred == y_true)

    def gradient(self, x, y_true):
        self.loss(x, y_true)  # 前向传播（训练模式）
        dout = (self.prob - y_true) / x.shape[0]

        # 反向传播
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        # 收集梯度
        grads = {}
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                grads[f'{layer.name}_W'] = layer.dW
                grads[f'{layer.name}_b'] = layer.db
            elif isinstance(layer, Dense):
                grads[f'{layer.name}_W'] = layer.dW
                grads[f'{layer.name}_b'] = layer.db
            elif isinstance(layer, BatchNorm):
                grads[f'{layer.name}_gamma'] = layer.dgamma
                grads[f'{layer.name}_beta'] = layer.dbeta
        return grads

    def save_params(self, path):
        params = {}
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                params[f'{layer.name}_W'] = layer.W
                params[f'{layer.name}_b'] = layer.b
            elif isinstance(layer, Dense):
                params[f'{layer.name}_W'] = layer.W
                params[f'{layer.name}_b'] = layer.b
            elif isinstance(layer, BatchNorm):
                params[f'{layer.name}_gamma'] = layer.gamma
                params[f'{layer.name}_beta'] = layer.beta
        np.savez(path, **params)

    def load_params(self, path):
        data = np.load(path)
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                layer.W = data[f'{layer.name}_W']
                layer.b = data[f'{layer.name}_b']
            elif isinstance(layer, Dense):
                layer.W = data[f'{layer.name}_W']
                layer.b = data[f'{layer.name}_b']
            elif isinstance(layer, BatchNorm):
                layer.gamma = data[f'{layer.name}_gamma']
                layer.beta = data[f'{layer.name}_beta']
