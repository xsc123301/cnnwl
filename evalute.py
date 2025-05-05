import numpy as np
from moudel.cnn import CNN
from dataset import load_cifar10

def evaluate():
    # 加载数据
    _, _, X_test, y_test = load_cifar10('E:/sy4/tuxianfenge/cifar-10-batches-py', augment=False)
    X_test = X_test.transpose(0, 3, 1, 2)  # 转NCHW

    # 加载模型
    model = CNN(use_dropout=False)  # 测试时关闭Dropout
    model.load_params('E:/sy4/tuxianfenge/model_params.npz')

    # 全量测试集评估
    test_acc = model.accuracy(X_test, y_test)
    print(f"测试集准确率: {test_acc:.4f}")

if __name__ == "__main__":
    evaluate()