import numpy as np
from moudel.cnn import CNN
from dataset import load_cifar10
from common.util import AdamOptimizer, plot_training_curve


# 超参数配置
BATCH_SIZE = 64
EPOCHS = 10
INIT_LR = 0.001
WEIGHT_DECAY = 1e-4

def train():
    # 加载数据（启用增强）
    X_train, y_train, X_test, y_test = load_cifar10('E:/sy4/tuxianfenge/cifar-10-batches-py', augment=True)
    X_train = X_train.transpose(0, 3, 1, 2)  # 转NCHW格式
    X_test = X_test.transpose(0, 3, 1, 2)

    # 初始化模型和优化器
    model = CNN(use_dropout=True)
    optimizer = AdamOptimizer(model.params, lr=INIT_LR, weight_decay=WEIGHT_DECAY)

    # 训练日志
    train_losses, val_accuracies = [], []
    lr = INIT_LR

    for epoch in range(EPOCHS):
        # 学习率衰减（每15轮减半）
        if epoch % 15 == 0 and epoch != 0:
            lr *= 0.5
            optimizer.lr = lr

        # 打乱数据
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0
        for i in range(0, len(X_train), BATCH_SIZE):
            X_batch = X_train[i:i+BATCH_SIZE]
            y_batch = y_train[i:i+BATCH_SIZE]

            # 前向传播与反向传播
            grads = model.gradient(X_batch, y_batch)
            optimizer.step(model.params, grads)

            # 计算损失
            loss = model.loss(X_batch, y_batch)
            epoch_loss += loss

        # 验证集评估
        val_acc = model.accuracy(X_test[:1000], y_test[:1000])
        train_losses.append(epoch_loss / (len(X_train)//BATCH_SIZE))
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_losses[-1]:.4f}, Val Acc: {val_acc:.4f}")

    # 保存模型与训练曲线
    model.save_params('model_params.npz')
    plot_training_curve(train_losses, val_accuracies)

if __name__ == "__main__":
    train()