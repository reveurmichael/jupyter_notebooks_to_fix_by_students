#!/usr/bin/env python
# 完整测试 PyTorch 神经网络 Notebook

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time

print("=" * 60)
print("PyTorch 神经网络 - 完整测试")
print("=" * 60)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n使用设备：{device}')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. 加载数据
# ============================================================
print('\n[1/6] 加载 MNIST 数据...')
start_time = time.time()

train_df = pd.read_csv('mnist_train.csv')
test_df = pd.read_csv('mnist_test.csv')

x_train = train_df.iloc[:, 1:].values.astype(np.float32) / 255.0
y_train = train_df.iloc[:, 0].values.astype(np.int64)
x_test = test_df.iloc[:, 1:].values.astype(np.float32) / 255.0
y_test = test_df.iloc[:, 0].values.astype(np.int64)

print(f'  训练数据：{x_train.shape}, 标签：{y_train.shape}')
print(f'  测试数据：{x_test.shape}, 标签：{y_test.shape}')
print(f'  加载耗时：{time.time() - start_time:.2f}s')

# 转换为张量
x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train)
x_test_tensor = torch.from_numpy(x_test)
y_test_tensor = torch.from_numpy(y_test)

# 创建 DataLoader
batch_size = 64
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f'  训练批次数：{len(train_loader)}')
print(f'  测试批次数：{len(test_loader)}')

# ============================================================
# 2. 创建模型
# ============================================================
print('\n[2/6] 创建 MLP 模型...')

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

model = MLP().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f'  模型参数量：{total_params:,}')

# ============================================================
# 3. 定义损失函数和优化器
# ============================================================
print('\n[3/6] 配置训练...')
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f'  损失函数：CrossEntropyLoss')
print(f'  优化器：Adam (lr={learning_rate})')

# ============================================================
# 4. 训练模型
# ============================================================
print('\n[4/6] 开始训练...')
num_epochs = 20
train_losses = []
train_accs = []

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    
    print(f'  Epoch [{epoch+1:2d}/{num_epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%')

training_time = time.time() - start_time
print(f'\n  训练完成！总耗时：{training_time:.1f}s ({training_time/60:.1f} 分钟)')

# ============================================================
# 5. 评估模型
# ============================================================
print('\n[5/6] 在测试集上评估...')

model.eval()
correct = 0
total = 0
test_loss = 0.0

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.numpy())

test_accuracy = 100.0 * correct / total
avg_test_loss = test_loss / len(test_loader)

print(f'  测试准确率：{test_accuracy:.2f}%')
print(f'  测试损失：{avg_test_loss:.4f}')
print(f'  正确预测：{correct}/{total}')

# ============================================================
# 6. 可视化结果
# ============================================================
print('\n[6/6] 生成可视化图表...')

# 训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accs, 'r-', label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print('  ✓ 训练曲线已保存：training_curves.png')

# 混淆矩阵
plt.figure(figsize=(10, 8))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=150)
print('  ✓ 混淆矩阵已保存：confusion_matrix.png')

# 分类报告
print('\n  分类报告:')
report = classification_report(all_labels, all_preds, digits=4)
print(report)

# ============================================================
# 总结
# ============================================================
print('\n' + '=' * 60)
print('✅ 完整测试通过！')
print('=' * 60)
print(f'\n最终结果:')
print(f'  训练准确率：{train_accs[-1]:.2f}%')
print(f'  测试准确率：{test_accuracy:.2f}%')
print(f'  训练时间：{training_time:.1f}s')
print(f'\n生成的文件:')
print(f'  - training_curves.png')
print(f'  - confusion_matrix.png')
print('\nNotebook 已准备就绪，可以提交 Pull Request！')
