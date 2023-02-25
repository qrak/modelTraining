import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from classdirectory.classfile import LSTMNet
from classdirectory.classfile import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

if __name__ == '__main__':
    """x_train = np.random.rand(100, 14)
    y_train = np.random.rand(100)
    x_val = np.random.rand(50, 14)
    y_val = np.random.rand(50)
    x_test = np.random.rand(50, 14)
    y_test = np.random.rand(50)"""
    # Load data into dataframe

    if torch.cuda.is_available():
        print('CUDA is available.')
    else:
        print('CUDA is not available.')
    """df = pd.read_csv('BTC_USDT_5m_2015-02-01_now_binance.csv', header=0, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # Calculate MACD
    df.ta.macd(append=True)
    # Calculate RSI30
    df.ta.rsi(append=True, length=14)
    # Calculate MOM30
    df.ta.mom(append=True, length=14)
    df.ta.adx(length=14, append=True)
    df.ta.cci(length=20, constant=0.015, append=True)
    sliced_rows = 50
    df = df.iloc[sliced_rows:]"""


    # Save the DataFrame to a CSV file
    #df = df.to_csv('BTC_USDT_5m_indicators.csv', index=False)
    df = pd.read_csv('BTC_USDT_5m_indicators.csv')


    scaler = StandardScaler()
    x = df[['open', 'high', 'low', 'close', 'volume', 'MACD_12_26_9', 'MACDs_12_26_9', 'RSI_14', 'MOM_14', 'ADX_14', 'DMP_14',
            'DMN_14', 'CCI_20_0.015']].values
    y = df['close'].values
    x = scaler.fit_transform(x)

    # Split the data into training, validation, and test sets
    x_train, x_valtest, y_train, y_valtest = train_test_split(x, y, test_size=0.3, shuffle=False)
    x_val, x_test, y_val, y_test = train_test_split(x_valtest, y_valtest, test_size=0.5, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create the training, validation, and test datasets
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32).to(device),
                                  torch.tensor(y_train, dtype=torch.float32).to(device))
    val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32).to(device),
                                torch.tensor(y_val, dtype=torch.float32).to(device))
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32).to(device),
                                 torch.tensor(y_test, dtype=torch.float32).to(device))

    # Create the data loaders for the training and validation sets
    batch_size = 64
    #batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    # Create the early stopping callback
    early_stopping = EarlyStopping(patience=10)
    # Create the model, loss function, and optimizer
    input_size = 13
    hidden_size = 32
    num_layers = 8
    output_size = 1
    dropout_size = 0.2
    model = LSTMNet(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout_size).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Initialize the early stopping callback

    # Train the model
    num_epochs = 150
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs,
                             labels.view(-1, 1).to(device))  # Move labels to the same device as the output tensor
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate the model on the validation set
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.unsqueeze(1))
                loss = criterion(outputs,
                                 labels.view(-1, 1).to(device))  # Move labels to the same device as the output tensor
                val_loss += loss.item()

        # Check if the validation loss has decreased, and if so, save the model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'saved_models/best_model.pt')

        # Call the EarlyStopping object
        early_stopping(val_loss, model)

        # If early stopping has been triggered, break out of the loop
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Evaluate the model on the test set
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.unsqueeze(1))
                loss = criterion(outputs,
                                 labels.view(-1, 1).to(device))  # Move labels to the same device as the output tensor
                test_loss += loss.item()

        # Print the training and validation loss for the epoch
        print(
            f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_loss / i:.6f} - Validation Loss: {val_loss / len(val_loader):.6f} - Test Loss: {test_loss / len(test_loader):.6f}')

    filename = f"model_{num_epochs}_{input_size}_{hidden_size}_{batch_size}_{num_layers}_{output_size}.pt"
    torch.save(model.state_dict(), filename)
    # Load the best model
    #model.load_state_dict(torch.load('saved_models/best_model.pt'))
