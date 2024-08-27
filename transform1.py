import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.data import CustomDataset, ToTensor

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=input_dim, nhead=1)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x[-1])  # Use the last output for prediction
        return x


# 训练函数
def train_model(model, train_loader, optimizer, criterion, num_epochs=100):



    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch['x'], batch['y']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader)}")


# Testing function
def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch['x'], batch['y']
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()

    average_loss = total_loss / len(test_loader)
    print(f"Test Loss: {average_loss}")
    return average_loss


model = TransformerModel(input_dim=2, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


# Example usage:
# Assuming `test_loader`, `model`, and `criterion` are already defined from the previous code snippets
# test_model(model, test_loader, criterion)
def train():
    # 准备数据
    x_train = [
        [[1.6, 1.6], [2, 1], [6.3, 1.1]],
        [[2, 1], [6.3, 1.1], [1.2, 1.3]],
        [[6.3, 1.1], [1.2, 1.3], [5.1, 1.9]]
    ]
    y_train = [1.2, 5.1, 7.9]
    # 转换数据为 Dataset
    train_dataset = CustomDataset(x_train, y_train, transform=ToTensor())
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # 初始化模型、优化器和损失函数
    # 训练模型
    train_model(model, train_loader, optimizer, criterion)

    # x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    # y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    # num_epochs = 100
    # for epoch in range(num_epochs):  # Number of epochs
    #     model.train()
    #     optimizer.zero_grad()
    #     # Forward pass
    #     outputs = model(x_train_tensor)
    #     loss = criterion(outputs, y_train_tensor)
    #     # Backward pass
    #     loss.backward()
    #     optimizer.step()


def test():
    x_test = [
        [[1.2, 1.3], [5.1, 1.9], [3, 5.8]],
        [[5.1, 1.9], [3, 5.8], [1.6, 1.6]],
    ]
    y_test = [1.6, 2.0]
    test_dataset = CustomDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1)
    test_model(model, test_loader, criterion)


if __name__ == '__main__':
    train()
    test()
