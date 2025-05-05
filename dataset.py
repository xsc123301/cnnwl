import numpy as np
import pickle
import os
from scipy.ndimage import rotate

def load_cifar10(path, augment=False):
    def _unpickle(file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
        return data

    # 加载训练数据
    X_train, y_train = [], []
    for i in range(1, 6):
        data = _unpickle(os.path.join(path, f'data_batch_{i}'))
        X_train.append(data['data'])
        y_train.extend(data['labels'])
    X_train = np.concatenate(X_train).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_train = np.array(y_train)

    # 加载测试数据
    test_data = _unpickle(os.path.join(path, 'test_batch'))
    X_test = test_data['data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_test = np.array(test_data['labels'])

    # 调整数据格式为NHWC
    X_train = X_train.transpose(0, 2, 3, 1)  # (N, H, W, C)
    X_test = X_test.transpose(0, 2, 3, 1)

    # 数据增强（仅训练集）
    if augment:
        X_train = np.array([_augment_image(x) for x in X_train])

    # One-hot编码标签
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    return X_train, y_train, X_test, y_test

def _augment_image(image):
    # 随机水平翻转
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
    # 随机裁剪（32x32 → 28x28 → 32x32）
    if np.random.rand() > 0.5:
        h, w = 28, 28
        top = np.random.randint(0, 32 - h)
        left = np.random.randint(0, 32 - w)
        image = image[top:top+h, left:left+w]
        image = np.pad(image, [(2,2), (2,2), (0,0)], mode='constant')
    # 随机旋转（±15度）
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-15, 15)
        image = rotate(image, angle, reshape=False, mode='reflect')
    # 颜色抖动
    if np.random.rand() > 0.5:
        image = image * np.random.uniform(0.8, 1.2)
        image = np.clip(image, 0, 1)
    return image

