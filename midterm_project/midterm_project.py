import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 数据预处理
def preprocess_data(data):
    # 将 'Type' 列映射为数值
    data['Type'] = data['Type'].map({'M': 1, 'L': 0, 'H': 2})
    
    # 删除无关的列
    data = data.drop(['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
    
    # 特征选择：除了 'Machine failure' 列之外的所有特征
    X = data.drop(columns=['Machine failure'])
    y = data['Machine failure']
    
    print("原始正类样本数量:", np.sum(y == 1))
    print("原始负类样本数量:", np.sum(y == 0))
    
    # 分离正类和负类
    X_negative = X[y == 0].reset_index(drop=True)
    y_negative = y[y == 0].reset_index(drop=True)
    X_positive = X[y == 1].reset_index(drop=True)
    y_positive = y[y == 1].reset_index(drop=True)
    
    # 随机选取 4% 的负类
    negative_sample_size = max(1, int(0.04 * len(X_negative)))  # 至少采样 1 个负类
    if negative_sample_size > len(X_negative):
        raise ValueError("负类数据不足，无法采样 4%")
    negative_indices = np.random.choice(len(X_negative), size=negative_sample_size, replace=False)
    X_negative_sampled = X_negative.iloc[negative_indices].to_numpy()
    y_negative_sampled = y_negative.iloc[negative_indices].values  # 使用 .values 确保索引有效
    
    # 合并正类和抽样后的负类
    X_sampled = np.vstack((X_positive.to_numpy(), X_negative_sampled))
    y_sampled = np.hstack((y_positive.to_numpy(), y_negative_sampled))
    
    # 打乱数据顺序
    indices = np.arange(len(y_sampled))
    np.random.shuffle(indices)
    X = X_sampled[indices]
    y = y_sampled[indices]
    
    print("采样后正类样本数量:", np.sum(y == 1))
    print("采样后负类样本数量:", np.sum(y == 0))
    
    # 特征归一化（最小-最大缩放）
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  # 归一化处理
    
    return X, y  # 返回 numpy 数组

# 划分数据集为训练集和测试集
def train_test_split(X, y, test_size=0.3, random_seed=42):
    np.random.seed(random_seed)  # 设置随机种子，保证结果可复现
    indices = np.arange(X.shape[0])  # 获取数据的索引
    np.random.shuffle(indices)  # 随机打乱索引
    split_index = int((1 - test_size) * len(indices))  # 计算训练集大小
    train_idx, test_idx = indices[:split_index], indices[split_index:]  # 划分训练集和测试集
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# 线性回归（用于二分类）
class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, batch_size=32):
        self.learning_rate = learning_rate  # 学习率
        self.epochs = epochs  # 迭代次数
        self.batch_size = batch_size  # 批量大小
        self.loss_history = []  # 存储每轮的损失
    
    def fit(self, X, y):
        m, n = X.shape  # 获取数据的行数（样本数）和列数（特征数）
        self.weights = np.zeros(n)  # 初始化权重为零
        self.bias = 0  # 初始化偏置为零
        
        for epoch in range(self.epochs):  # 迭代训练
            epoch_loss = 0  # 初始化每轮损失
            # 打乱数据集
            indices = np.arange(m)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
    
            # Mini-Batch Gradient Descent
            for start in range(0, m, self.batch_size):
                end = start + self.batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]
    
                # 前向计算
                y_pred = np.dot(X_batch, self.weights) + self.bias  # 预测值
                error = y_pred - y_batch  # 计算误差
    
                # 损失计算（MSE）
                loss = np.mean(error ** 2)
                epoch_loss += loss * len(y_batch)  # 累积损失
    
                # 梯度计算
                dW = (2 / self.batch_size) * np.dot(X_batch.T, error)  # 权重梯度
                db = (2 / self.batch_size) * np.sum(error)  # 偏置梯度
    
                # 参数更新
                self.weights -= self.learning_rate * dW  # 更新权重
                self.bias -= self.learning_rate * db  # 更新偏置
            
            # 计算平均损失
            epoch_loss /= m
            self.loss_history.append(epoch_loss)
    
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias  # 计算预测值
        return (y_pred > 0.5).astype(int)  # 如果预测值大于0.5，预测为1，否则为0

# 感知机
class Perceptron:
    def __init__(self, learning_rate=0.001, epochs=1000, batch_size=128):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_history = []  # 存储每轮的损失
    
    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        y = np.where(y > 0, 1, -1)  # 将标签转换为 -1/1
        learning_rate = self.learning_rate  # 初始化学习率

        # 特征标准化
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        for epoch in range(self.epochs):
            misclassifications = 0
            indices = np.arange(m)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for start in range(0, m, self.batch_size):
                end = start + self.batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]
                linear_output = np.dot(X_batch, self.weights) + self.bias
                predictions = np.sign(linear_output)

                # 计算误分类点
                misclassified = y_batch * linear_output <= 0
                num_misclassified = np.sum(misclassified)
                misclassifications += num_misclassified

                # 更新权重和偏置
                if np.any(misclassified):
                    gradient_w = np.sum(
                        learning_rate * y_batch[misclassified, None] * X_batch[misclassified], axis=0
                    ) - learning_rate * 0.01 * self.weights  # L2 正则化
                    gradient_b = np.sum(learning_rate * y_batch[misclassified])

                    self.weights += gradient_w
                    self.bias += gradient_b

            # 动态调整学习率
            learning_rate /= (1 + 0.1 * epoch)
            self.loss_history.append(misclassifications / m)  # 记录损失

    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return (linear_output > 0).astype(int)

# 逻辑回归
class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000, batch_size=8):
        self.learning_rate = learning_rate  # 学习率
        self.epochs = epochs  # 训练轮数
        self.batch_size = batch_size  # 每个批次的大小
        self.loss_history = []  # 存储每轮的损失
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  # 限制 z 的范围，避免 np.exp(-z) 溢出
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)  # 初始化权重
        self.bias = 0  # 初始化偏置
    
        for epoch in range(self.epochs):
            epoch_loss = 0  # 初始化每轮损失
            # 打乱数据顺序
            indices = np.arange(m)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            
            # Mini-Batch 训练
            for start in range(0, m, self.batch_size):
                end = start + self.batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]
                
                # 前向传播
                linear_output = np.dot(X_batch, self.weights) + self.bias
                y_pred = self.sigmoid(linear_output)
                
                # 损失计算（二元交叉熵）
                epsilon = 1e-15  # 防止 log(0)
                y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
                loss = -np.mean(y_batch * np.log(y_pred) + (1 - y_batch) * np.log(1 - y_pred))
                epoch_loss += loss * len(y_batch)  # 累积损失
                
                # 计算误差
                error = y_pred - y_batch
    
                # 梯度下降更新
                dW = np.dot(X_batch.T, error) / self.batch_size
                db = np.sum(error) / self.batch_size
    
                self.weights -= self.learning_rate * dW  # 更新权重
                self.bias -= self.learning_rate * db  # 更新偏置
            
            # 计算平均损失
            epoch_loss /= m
            self.loss_history.append(epoch_loss)
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_output)
        return (y_pred > 0.5).astype(int)

# 多层感知机
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros(hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros(output_size)
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        n_samples = X.shape[0]

        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            epoch_loss = 0
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]

                z1 = np.dot(X_batch, self.weights1) + self.bias1
                a1 = self.sigmoid(z1)
                z2 = np.dot(a1, self.weights2) + self.bias2
                a2 = self.sigmoid(z2)

                error = a2 - y_batch
                epoch_loss += np.mean(np.square(error))

                d_output = error * self.sigmoid_derivative(a2)
                d_hidden = np.dot(d_output, self.weights2.T) * self.sigmoid_derivative(a1)

                self.weights2 -= self.learning_rate * np.dot(a1.T, d_output)
                self.bias2 -= self.learning_rate * np.sum(d_output, axis=0)
                self.weights1 -= self.learning_rate * np.dot(X_batch.T, d_hidden)
                self.bias1 -= self.learning_rate * np.sum(d_hidden, axis=0)

            self.loss_history.append(epoch_loss / (n_samples // self.batch_size))

    def predict(self, X):
        z1 = np.dot(X, self.weights1) + self.bias1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.weights2) + self.bias2
        a2 = self.sigmoid(z2)
        return (a2 > 0.5).astype(int).flatten()


# 评估指标
def evaluate(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)  # 准确率
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
    recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1

# 主程序
if __name__ == "__main__":
    # 加载数据集（确保数据集的路径正确）
    data = pd.read_csv('ai4i2020.csv')
    
    # 数据预处理
    X, y = preprocess_data(data)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 训练和评估模型
    models = {
        "线性回归": LinearRegression(),
        "感知机": Perceptron(),
        "逻辑回归": LogisticRegression(),
        "多层感知机(MLP)": MLP(input_size=X_train.shape[1], hidden_size=10, output_size=1)
    }
    
    # 存储每个模型的损失历史
    loss_histories = {}
    
    for name, model in models.items():
        print(f"\n正在训练模型: {name}")
        model.fit(X_train, y_train)  # 训练模型
        y_pred = model.predict(X_test)  # 预测
        accuracy, precision, recall, f1 = evaluate(y_test, y_pred)  # 计算评估指标
        print(f"{name}: 准确率={accuracy:.2f}, 精确率={precision:.2f}, 召回率={recall:.2f}, F1分数={f1:.2f}")
        loss_histories[name] = model.loss_history  # 记录损失历史
    
    # 可视化不同模型的损失函数变化
    plt.figure(figsize=(10, 6))
    for name, loss_history in loss_histories.items():
        plt.plot(loss_history, label=name)
    plt.title('loss function of different models')
    plt.xlabel('number of training (Epochs)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
