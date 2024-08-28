import torch
import torch.nn as nn
import torch.optim as optim
import math


class GPTPredictor(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, hidden_size=16, num_heads=2, dropout=0.1):
        super(GPTPredictor, self).__init__()
        self.input_embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)
        decoder_layer = nn.TransformerDecoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, y_history):
        # Embedding and positional encoding for input x
        embedded_x = self.input_embedding(x)
        embedded_x = embedded_x.transpose(0, 1)  # (batch_size, seq_len, hidden_size)
        embedded_x = self.positional_encoding(embedded_x)

        # Embedding and positional encoding for history y
        embedded_history = self.input_embedding(y_history)
        embedded_history = embedded_history.transpose(0, 1)  # (batch_size, seq_len, hidden_size)
        embedded_history = self.positional_encoding(embedded_history)

        # Transformer decoder
        memory = embedded_history[:-1]  # Use history as memory, excluding the last token
        tgt = embedded_history[-1:]  # Predict the next token based on last token in history

        output = self.transformer_decoder(tgt, memory)

        # Predict next value
        output = self.fc(output)  # No need to take the output of the last transformer layer

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# Example usage
input_size = 5  # Number of unique elements in input x (1, 2, 3, 4, and 5)
output_size = 5  # Predicting the next value from the same set of inputs
model = GPTPredictor(input_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (for demonstration)
x = torch.tensor([1, 2, 3, 4])  # Input sequence
y_history = torch.tensor([1, 2, 3, 4])  # History of observed values
y_true = torch.tensor([5])  # Target next value
epochs = 1000

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x.unsqueeze(0), y_history.unsqueeze(0))  # Model prediction
    loss = criterion(y_pred.squeeze(0), y_true)  # Squeeze the batch dimension for loss calculation
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')

# After training, you can make predictions
model.eval()
with torch.no_grad():
    predicted_probs = model(x.unsqueeze(0), y_history.unsqueeze(0))
    _, predicted_class = predicted_probs.max(dim=2)
    print(f'Predicted next value: {predicted_class.item()}')