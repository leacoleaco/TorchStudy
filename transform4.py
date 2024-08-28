import math

import torch
import torch.nn as nn
import torch.optim as optim


class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, hidden_size=16, num_heads=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Embedding and positional encoding
        embedded = self.input_embedding(x)
        embedded = embedded.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        embedded = self.positional_encoding(embedded)

        # Transformer encoder
        encoded = self.transformer_encoder(embedded)

        # Decode to output size
        decoded = self.decoder(encoded[-1])  # Taking the output of the last transformer layer

        return decoded


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


def main():
    # Example usage
    input_size = 5  # Number of unique elements in input x (1, 2, 3, 4, and 5)
    output_size = 1  # Predicting a single value y
    model = TransformerModel(input_size, output_size)
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop (for demonstration)
    x = torch.tensor([1, 2, 3, 4])  # Input sequence
    y_true = torch.tensor([5.0])  # Target output
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_input = x.unsqueeze(1)
        y_pred = model(x_input)  # Model prediction
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')
    # After training, you can make predictions
    model.eval()
    with torch.no_grad():
        predicted_output = model(x.unsqueeze(1))
        print(f'Predicted output: {predicted_output[0]}')


if __name__ == '__main__':
    main()
