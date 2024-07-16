# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate synthetic data for example
def generate_synthetic_data(seq_length=10, num_sequences=1000):
    data = []
    for _ in range(num_sequences):
        sequence = np.random.rand(seq_length, 1)
        target = np.sum(sequence)
        data.append((sequence, target))
    return data

# Define an RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def train_model():
    # Hyperparameters
    input_size = 1
    hidden_size = 50
    output_size = 1
    learning_rate = 0.01
    num_epochs = 20

    # Generate synthetic data
    data = generate_synthetic_data()
    train_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = RNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for sequences, targets in train_loader:
            sequences = torch.tensor(sequences, dtype=torch.float32)
            targets = torch.tensor(targets, dtype=torch.float32)

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the model
    torch.save(model.state_dict(), './model/rnn_model.pth')
    print('Finished Training')

if __name__ == "__main__":
    train_model()
