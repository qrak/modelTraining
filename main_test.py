import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and preprocess the data
data = pd.read_csv('BTC_USDT_5m_with_indicators2.csv', parse_dates=['timestamp'])
data = data.sort_values('timestamp')
data = data.reset_index(drop=True)
data = data[['open', 'high', 'low', 'close', 'volume']]
scaler = StandardScaler()
data = scaler.fit_transform(data)
X = data[:-1]
y = data[1:, 3]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale the target variable separately
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# Define the neural network architecture
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the hyperparameters
input_size = X_train.shape[1]
hidden_size = 32
output_size = 1
learning_rate = 1e-3
num_epochs = 50
batch_size = 32

# Define the model, loss function, and optimizer
model = FFNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    loss = 0.0
    for i in range(0, X_train.shape[0], batch_size):
        optimizer.zero_grad()
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        y_hat = model(X_batch)
        batch_loss = criterion(y_hat.squeeze(), y_batch)
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item() * X_batch.shape[0]
    loss /= X_train.shape[0]

    # Evaluate the model on the validation set
    with torch.no_grad():
        y_val_hat = model(X_val)
        val_loss = criterion(y_val_hat.squeeze(), y_val)
        print(f'Epoch {epoch+1}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')

# Evaluate the model on the test set
with torch.no_grad():
    X_test = torch.tensor(data[-1:, :-1], dtype=torch.float32)
    y_test = torch.tensor(data[-1:, -1:], dtype=torch.float32)
    y_test_hat = model(X_test)
    y_test_hat = y_scaler.inverse_transform(y_test_hat.numpy().reshape(-1, 1))
    y_test = y_scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
    test_loss = criterion(torch.tensor(y_test_hat), y_test)
    print(f'Test Loss: {test_loss:.4f}')

