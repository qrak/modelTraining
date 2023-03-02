import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, TensorDataset
from classdirectory.classfile_test import LSTMNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    data = pd.read_csv('BTC_USDT_1h_with_indicators.csv', parse_dates=['timestamp'])
    # Select the first 100 rows of the DataFrame
    scaler = StandardScaler()
    X = data.drop(['timestamp', 'close'], axis=1)
    y = data[['close']]
    Xy = scaler.fit_transform(pd.concat([X, y], axis=1))

    X_train_val, X_test, y_train_val, y_test = train_test_split(Xy[:, :-1], Xy[:, -1], test_size=0.2, shuffle=False,
                                                                random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False,
                                                      random_state=42)

    # Define the hyperparameters to search over
    input_size = (X.shape[1])
    hidden_size = 32
    num_layers = 4
    dropout_size = 0.1
    torch.set_float32_matmul_precision('high')

    print(
        f"Loading model with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout_size}")

    # Create the PyTorch datasets and data loaders
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).to(device),
                                 torch.tensor(y_test, dtype=torch.float32).to(device))
    test_data_loader = DataLoader(test_dataset, batch_size=96, shuffle=False)

    # Load the best model
    best_model_path = f"save/best_model_{input_size}_{hidden_size}_{num_layers}_{dropout_size}.pt"
    best_model = LSTMNet(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=1,
                         dropout=dropout_size)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(device)

    if torch.cuda.is_available():
        best_model.to('cuda')

    # Define a new variable to store the predicted values
    y_test_pred = []
    # Iterate over the test data and predict the output
    best_model.eval()
    with torch.no_grad():
        for X_test_batch, _ in test_data_loader:
            # Move the input tensor to the same device as the model
            if torch.cuda.is_available():
                X_test_batch = X_test_batch.to('cuda', non_blocking=True)
            y_test_batch_pred = best_model(X_test_batch)
            if torch.cuda.is_available():
                y_test_batch_pred = y_test_batch_pred.cpu()
            y_test_pred.append(y_test_batch_pred.numpy())
    y_test_pred = np.concatenate(y_test_pred)

    # Create a new StandardScaler object and fit it on the un-scaled data
    scaler = StandardScaler()
    scaler.fit(data[["close"]].values)

    # Inverse transform the predicted values using the new scaler object
    y_test_pred = scaler.inverse_transform(y_test_pred)

    # Print the last 10 values of 'close' and 'predicted close'
    for i in range(-10, 0):
        print(f"Actual close: {data.iloc[i]['close']:.2f}, Predicted close: {y_test_pred[i][0]:.2f}")
