import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
from classdirectory.classfile import LSTMNet
from classdirectory.classfile import EarlyStopping
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from torch.utils.data import DataLoader, TensorDataset

if __name__ == '__main__':

    # Load data into dataframe

    if torch.cuda.is_available():
        print('CUDA is available.')
    else:
        print('CUDA is not available.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    #df = df.to_csv('test_df.csv', index=False)
    #df = pd.read_csv('BTC_USDT_5m_with_indicators.csv')
    df = pd.read_csv('BTC_USDT_5m_with_indicators.csv')
    df = df.dropna()
    scaler = StandardScaler()
    X = df.drop(['close', 'timestamp'], axis=1).values
    y = df['close'].values
    X = scaler.fit_transform(X)
    # Perform feature selection
    k = 10  # Number of features to select
    selector = SelectKBest(f_regression, k=k)
    X = selector.fit_transform(X, y)
    # Split the data into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, shuffle=False)

    # Define the number of folds for cross-validation
    num_folds = 5

    # Define the hyperparameters to search over
    input_sizes = [X.shape[1]]
    hidden_sizes = [16, 32, 64]
    num_layers_list = [2, 4, 6]
    dropout_sizes = [0.1, 0.2]

    num_epochs = 100
    best_val_loss = float('inf')
    best_model = None
    model = None
    train_loader = None
    optimizer = None
    criterion = None
    val_loader = None
    best_hyperparams = None
    for input_size in input_sizes:
        for hidden_size in hidden_sizes:
            for num_layers in num_layers_list:
                for dropout_size in dropout_sizes:
                    print(
                        f"Training model with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout_size}")
                    val_losses = []
                    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
                    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
                        print(f"Fold {fold + 1}/{num_folds}")
                        # Split the data into training and validation sets for this fold
                        X_fold_train, y_fold_train = X_train_val[train_idx], y_train_val[train_idx]
                        X_fold_val, y_fold_val = X_train_val[val_idx], y_train_val[val_idx]

                        # Create the PyTorch datasets and data loaders
                        train_dataset = TensorDataset(torch.tensor(X_fold_train, dtype=torch.float32),
                                                      torch.tensor(y_fold_train, dtype=torch.float32))
                        val_dataset = TensorDataset(torch.tensor(X_fold_val, dtype=torch.float32),
                                                    torch.tensor(y_fold_val, dtype=torch.float32))
                        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
                        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
                        # Create the model, loss function, and optimizer
                        model = LSTMNet(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                        output_size=1, dropout=dropout_size).to(device)
                        criterion = nn.MSELoss().to(device)
                        optimizer = optim.Adam(model.parameters(), lr=0.0001)
                        # Initialize the EarlyStopping callback
                        early_stopping = EarlyStopping(patience=10)

                        # Train the model
                        for epoch in range(num_epochs):
                            train_loss = 0.0
                            model.train()
                            for i, (inputs, labels) in enumerate(train_loader):
                                inputs, labels = inputs.to(device), labels.to(device)
                                optimizer.zero_grad()
                                outputs = model(inputs.unsqueeze(1))
                                loss = criterion(outputs, labels.view(-1, 1).to(
                                    device))  # Move labels to the same device as the output tensor
                                loss.backward()
                                optimizer.step()
                                train_loss += loss.item()

                            # Evaluate the model on the validation set
                            val_loss = model.evaluate(val_loader, criterion)

                            # Call the EarlyStopping callback
                            early_stopping(val_loss, model)

                            # If early stopping has been triggered, break out of the loop
                            if early_stopping.early_stop:
                                break
                            if (epoch + 1) % 10 == 0:
                                print(
                                    f"Fold {fold + 1}/{num_folds}, Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss / len(train_loader):.6f}, Validation Loss: {val_loss / len(val_loader):.6f}")

                        # Evaluate the model on the validation set for this fold
                        val_loss = model.evaluate(val_loader, criterion)
                        val_losses.append(val_loss)

                        mean_val_loss = np.mean(val_losses[:fold + 1])
                        print(f"Fold {fold + 1}/{num_folds}, Mean Validation Loss: {mean_val_loss:.6f}")

                        # Check if this model has a lower validation loss than the previous best model
                        if mean_val_loss < best_val_loss:
                            best_val_loss = mean_val_loss
                            best_model = model
                            best_hyperparams = (input_size, hidden_size, num_layers, dropout_size)

    # Train the best model on the full training set and evaluate it on the test set
    early_stopping = EarlyStopping(patience=10)
    for epoch in range(num_epochs):
        train_loss = 0.0
        best_model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = best_model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels.view(-1, 1).to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate the model on the validation set
        val_loss = best_model.evaluate(val_loader, criterion)

        # Call the early stopping callback
        early_stopping(val_loss, best_model)

        # If early stopping has been triggered, break out of the loop
        if early_stopping.early_stop:
            break

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss / len(train_loader):.6f}, Validation Loss: {val_loss / len(val_loader):.6f}")
    # Evaluate the best model on the test set
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).to(device),
                                 torch.tensor(y_test, dtype=torch.float32).to(device))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    test_loss = best_model.evaluate(test_loader, criterion)
    print(f"Test Loss: {test_loss:.6f}")

    print(f"Best hyperparameters: input_size={best_hyperparams[0]}, hidden_size={best_hyperparams[1]},num_layers={best_hyperparams[2]}, dropout={best_hyperparams[3]}")
    print(f"Validation loss: {best_val_loss:.6f}")
    print(f"Test loss: {test_loss / len(test_loader):.6f}")
    # Save the best model with the best hyperparameters
    torch.save(best_model.state_dict(), "best_modelor.pt")

