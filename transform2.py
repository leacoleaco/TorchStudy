import torch
import torch.nn as nn
import torch.optim as optim

# Prepare the data
x_train = torch.tensor([
    [[1, 1], [2, 2], [3, 3]],
    [[2, 2], [3, 3], [4, 4]],
    [[3, 3], [4, 4], [5, 5]],
], dtype=torch.float32)
y_train = torch.tensor([4, 5, 6], dtype=torch.float32).view(-1, 1)

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=input_dim, nhead=1, num_encoder_layers=1, num_decoder_layers=1)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x shape: (sequence_length, batch_size, input_dim)
        x = x.permute(1, 0, 2)  # Change to (batch_size, sequence_length, input_dim)
        x = self.transformer(x, x)
        x = x.mean(dim=0)  # Average over the sequence length
        return self.fc(x)

# Initialize model, loss function, and optimizer
model = TransformerModel(input_dim=2, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):  # Number of epochs
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(x_train)
    print("Predictions:", predictions.squeeze().numpy())
