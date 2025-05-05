import numpy as np
import matplotlib.pyplot as plt

class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in grads:
            # 添加L2正则化
            if '_W' in k or '_gamma' in k or '_beta' in k:
                grads[k] += self.weight_decay * params[k]
            # Adam更新
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

def plot_training_curve(train_loss, val_acc, save_path='training_curve.png'):
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def im2col(input_data, kernel_size, stride, pad):
    N, C, H, W = input_data.shape
    KH, KW = kernel_size, kernel_size
    H_out = (H + 2 * pad - KH) // stride + 1
    W_out = (W + 2 * pad - KW) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')
    col = np.zeros((N, C, KH, KW, H_out, W_out))

    for y in range(KH):
        y_max = y + stride * H_out
        for x in range(KW):
            x_max = x + stride * W_out
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)
    return col


def col2im(col, input_shape, kernel_size, stride, pad):
    N, C, H, W = input_shape
    KH, KW = kernel_size, kernel_size
    H_out = (H + 2 * pad - KH) // stride + 1
    W_out = (W + 2 * pad - KW) // stride + 1

    col_reshaped = col.reshape(N, H_out, W_out, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2 * pad, W + 2 * pad))

    for y in range(KH):
        for x in range(KW):
            img[:, :, y:y + H_out * stride:stride, x:x + W_out * stride:stride] += col_reshaped[:, :, y, x, :, :]

    if pad != 0:
        img = img[:, :, pad:-pad, pad:-pad]
    return img