import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.data import CustomDataset, ToTensor


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = int(d_model / num_heads)

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        attention_weights = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt \
            (torch.tensor(self.depth, dtype=torch.float32))
        attention_weights = torch.softmax(attention_weights, dim=-1)

        output = torch.matmul(attention_weights, V)
        output = self._combine_heads(output)

        output = self.W_O(output)
        return output

    def _split_heads(self, tensor):
        tensor = tensor.view(tensor.size(0), -1, self.num_heads, self.depth)
        return tensor.transpose(1, 2)

    def _combine_heads(self, tensor):
        tensor = tensor.transpose(1, 2).contiguous()
        tensor = tensor.view(tensor.size(0), -1, self.num_heads * self.depth)
        return tensor


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attention_output = self.attention(x, x, x)
        attention_output = self.norm1(x + attention_output)

        feedforward_output = self.feedforward(attention_output)
        output = self.norm2(attention_output + feedforward_output)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output):
        self_attention_output = self.self_attention(x, x, x)
        self_attention_output = self.norm1(x + self_attention_output)

        encoder_attention_output = self.encoder_attention(self_attention_output, encoder_output, encoder_output)
        encoder_attention_output = self.norm2(self_attention_output + encoder_attention_output)

        feedforward_output = self.feedforward(encoder_attention_output)
        output = self.norm3(encoder_attention_output + feedforward_output)
        return output


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.encoder_layers = nn.ModuleList([EncoderLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)

        # Encoder layers
        encoder_output = x.transpose(0, 1)
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)

        # Decoder layers
        # 编码器最后一层的输出作为解码器的输入，当然也可以使用所有时间步长，则去掉这一行，将整个encoder_output传递给解码器
        decoder_output = encoder_output[-1, :, :].unsqueeze(0)
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output)

        # Output layer
        output = self.output_layer(decoder_output.squeeze(0))
        return output


# 创建模型实例
model = Transformer(input_dim=2, hidden_dim=1568, num_heads=8, num_layers=60)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
device = "cuda"


def train():
    # 训练数据的维度为 (batch_size, 20, 2)，标签的维度为 (batch_size, 3)

    # x_train = [
    #     [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13],
    #      [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20]],
    #     [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13],
    #      [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20], [21, 21]],
    # ]
    # y_train = [[0, 0, 1], [1, 0, 0]]
    # # 转换数据为 Dataset
    # train_dataset = CustomDataset(x_train, y_train, transform=ToTensor())
    # # 创建 DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1, shuffle=True, )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True, )

    # 开始训练
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # 输入数据
            # data, target = data.to(device), target.to(device)
            # inputs, labels = data
            # inputs = data['x']
            # labels = data['y']

            inputs = data.view(1, 784).unsqueeze(2).repeat(1, 1, 2)
            labels = F.one_hot(target, num_classes=10)

            optimizer.zero_grad()
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计损失
            running_loss += loss.item()

        # 输出统计信息
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, running_loss / len(train_loader)))


def test():
    x_test = [
        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13],
         [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20]],
    ]
    y_test = [[0, 0, 1]]
    test_dataset = CustomDataset(x_test, y_test, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1)

    # 测试模式
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch['x'], batch['y']
            outputs = model(inputs)
            predict = outputs
            loss = criterion(predict, targets)
            total_loss += loss.item()

    average_loss = total_loss / len(test_loader)
    print(f"Test Loss: {average_loss}")
    return average_loss


if __name__ == '__main__':
    train()
    test()
