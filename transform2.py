import torch
import torch.nn as nn
import torch.optim as optim


# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=input_dim, nhead=1, batch_first=True)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src, tgt):
        x = self.transformer(src, tgt)
        x = self.fc(x[:, -1, :])  # Use the last output for prediction
        return x


# Prepare the data
x_train = [
    [[1.6, 1.6], [2, 1], [6.3, 1.1]],
    [[2, 1], [6.3, 1.1], [1.2, 1.3]],
    [[6.3, 1.1], [1.2, 1.3], [5.1, 1.9]]
]
y_train = [1.2, 5.1, 7.9]

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Prepare target tensor (using y_train as the target)
# The target should have the same batch size as x_train
tgt_train_tensor = torch.zeros_like(x_train_tensor)  # Initialize with zeros
tgt_train_tensor[:, :-1, :] = x_train_tensor[:, 1:, :]  # Shift input for next step prediction
tgt_train_tensor[:, -1, :] = y_train_tensor  # Last step is the actual target

# Initialize model, loss function, and optimizer
model = TransformerModel(input_dim=2, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):  # Number of epochs
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(x_train_tensor, tgt_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass
    loss.backward()
    optimizer.step()

# Evaluation (optional)
model.eval()
with torch.no_grad():
    predictions = model(x_train_tensor, tgt_train_tensor)
    print(predictions)
