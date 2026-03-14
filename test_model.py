#!/usr/bin/env python
# 测试 PyTorch 模型是否能正常运行

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

print("=" * 50)
print("PyTorch 神经网络测试")
print("=" * 50)

# 检查版本
print(f'\nPyTorch 版本：{torch.__version__}')
print(f'Python 版本：{pd.__version__}')

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备：{device}')

# 定义简单的 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden1=256, hidden2=128, num_classes=10, dropout_rate=0.3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden2, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# 创建模型
model = MLP().to(device)
print(f'\n模型创建成功！')
print(f'总参数量：{sum(p.numel() for p in model.parameters()):,}')

# 加载数据测试
print('\n正在加载 MNIST 数据...')
train_df = pd.read_csv('mnist_train.csv', nrows=100)  # 只加载前 100 行测试，自动识别表头
x_train = train_df.iloc[:, 1:].values.astype(np.float32) / 255.0
y_train = train_df.iloc[:, 0].values.astype(np.int64)

print(f'训练数据形状：{x_train.shape}')

# 转换为张量
x_tensor = torch.from_numpy(x_train).to(device)
y_tensor = torch.from_numpy(y_train).to(device)

# 测试前向传播
print('\n测试前向传播...')
model.eval()
with torch.no_grad():
    outputs = model(x_tensor)
    _, predicted = outputs.max(1)
    accuracy = predicted.eq(y_tensor).sum().item() / len(y_tensor) * 100

print(f'测试准确率：{accuracy:.2f}% (未训练的随机模型)')

# 测试训练步骤
print('\n测试训练步骤...')
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 单次训练迭代
outputs = model(x_tensor)
loss = criterion(outputs, y_tensor)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f'初始损失：{loss.item():.4f}')

print('\n' + '=' * 50)
print('✅ 所有测试通过！Notebook 可以正常运行')
print('=' * 50)
