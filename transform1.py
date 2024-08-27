import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.data = x
        self.targets = y
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx], self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


# 定义自定义转换函数
class ToTensor(object):
    def __call__(self, sample):
        x, y = sample
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# 数据
x_train = [[1.6, 1.6], [2, 1], [6.3, 1.1]]
y_train = [2, 6.3, 7]
x_test = [[1.2, 1.3], [5.1, 1.9]]
y_test = [3, 5.8]

# 实例化训练集和测试集
train_dataset = CustomDataset(x_train, y_train, transform=transforms.Compose([ToTensor()]))
test_dataset = CustomDataset(x_test, y_test, transform=transforms.Compose([ToTensor()]))

# 使用DataLoader加载数据
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# 定义简单的神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(2, 1)  # 输入特征数为2，输出特征数为1

    def forward(self, x):
        x = self.fc(x)
        return x


# 实例化模型、损失函数和优化器
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# 训练模型
def train(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}")


# 测试模型
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))  # 计算损失
            test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader)}")


if __name__ == '__main__':
    # 训练和测试模型
    train(model, train_loader, optimizer, criterion, epochs=20)
    test(model, test_loader, criterion)
