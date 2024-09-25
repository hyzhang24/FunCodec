import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 随机生成一个小型数据集
np.random.seed(42)  # 设置随机种子以确保结果可重复

# 生成特征数据 X 和目标数据 y
n_samples = 100
X = np.random.rand(n_samples, 1) * 10  # 生成0到10之间的随机数
y = 2 * X + 1 + np.random.randn(n_samples, 1) * 2  # y = 2x + 1 + 噪声

# 转换为PyTorch张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 定义一个简单的三层全连接神经网络
model = nn.Sequential(
    nn.Linear(1, 10),  # 第一层：输入维度1，输出维度10
    nn.ReLU(),         # 激活函数
    nn.Linear(10, 20), # 第二层：输入维度10，输出维度20
    nn.ReLU(),         # 激活函数
    nn.Linear(20, 1)   # 第三层：输入维度20，输出维度1
)

# 初始化损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

# 训练模型
epochs = 1000
for epoch in range(epochs):
    # 前向传播
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 保存训练后的模型权重
torch.save(model.state_dict(), 'model_weights.pth')
print("Model weights saved to 'model_weights.pth'")