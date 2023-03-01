import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from classdirectory.classfile import LSTMNet
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from torch.utils.data import DataLoader, TensorDataset


if __name__ == '__main__':

    # Load data into dataframe
    data = pd.read_csv('BTC_USDT_5m_with_indicators2.csv', parse_dates=['timestamp'])
    #data = pd.read_csv('csv/BTC_USDT_1m_2022-01-01_now_binance.csv', parse_dates=['timestamp'])
    data = data.sort_values('timestamp')
    data = data.reset_index(drop=True)
    data = data[['open', 'high', 'low', 'close', 'volume']]
    data = data.dropna()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(how='any')

    scaler = StandardScaler()
    # Extract the features and target variable
    X = scaler.fit_transform(data.iloc[:, :-1])
    y = data.iloc[:, -1].values

    # Split the data into training and validation sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)

    # Scale the target variable separately
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # Move the data to the specified device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the number of folds for cross-validation
    num_folds = 2

    # Define the hyperparameters to search over
    input_sizes = [X.shape[1]]
    hidden_sizes = [32]
    num_layers_list = [2]
    dropout_sizes = [0.1]
    torch.set_float32_matmul_precision('high')
    num_epochs = 100
    best_val_loss = float('inf')
    best_model = LSTMNet(input_size=X.shape[1], hidden_size=hidden_sizes[0], num_layers=num_layers_list[0],
                                      output_size=1, dropout=dropout_sizes[0]).to(device)
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
                    best_fold_val_loss = float('inf')
                    best_val_losses = []
                    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                        print(f"Fold {fold + 1}/{num_folds}")
                        # Split the data into training and validation sets for this fold
                        X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
                        X_fold_val, y_fold_val = X_train[val_idx], y_train[val_idx]

                        # Create the PyTorch datasets and data loaders
                        train_dataset = TensorDataset(torch.tensor(X_fold_train, dtype=torch.float32).to(device),
                                                      torch.tensor(y_fold_train, dtype=torch.float32).to(device))
                        val_dataset = TensorDataset(torch.tensor(X_fold_val, dtype=torch.float32).to(device),
                                                    torch.tensor(y_fold_val, dtype=torch.float32).to(device))
                        train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=0,
                                                  pin_memory=False)
                        val_loader = DataLoader(val_dataset, batch_size=96, shuffle=False, num_workers=0,
                                                pin_memory=False)

                        model = LSTMNet(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                        output_size=1, dropout=dropout_size).to(device)

                        # Initialize the EarlyStopping callback
                        early_stopping = pl.callbacks.EarlyStopping(patience=20, monitor='val_loss')

                        # Train the model
                        trainer = pl.Trainer(max_epochs=num_epochs, accelerator='gpu', devices=1, callbacks=[early_stopping])
                        trainer.fit(model, train_loader, val_loader)

                        # Evaluate the model on the validation set for this fold
                        val_loss = trainer.validate(model, val_loader)[0]['val_loss']
                        val_losses.append(val_loss)

                        mean_val_loss = np.mean(val_losses[:fold + 1])
                        print(f"Fold {fold + 1}/{num_folds}, Mean Validation Loss: {mean_val_loss:.6f}")

                        # Check if this model has a lower validation loss than the previous best model
                        if val_loss < best_fold_val_loss:
                            # Save the model
                            model_dir = f"model_fold{fold + 1}_input_size={input_size}_hidden_size={hidden_size}_num_layers={num_layers}_dropout={dropout_size}"
                            os.makedirs(model_dir, exist_ok=True)
                            model_filename = os.path.join(model_dir, "best_model.pt")
                            torch.save(model.state_dict(), model_filename)
                            best_fold_val_loss = val_loss
                            best_val_losses.append(val_loss)
                            # Save the validation loss in the model directory
                            val_loss_filename = os.path.join(model_dir, "best_val_loss.txt")
                            with open(val_loss_filename, "w") as f:
                                f.write(str(val_loss))

                    # Check if this set of hyperparameters is the best so far
                    mean_val_loss = np.mean(best_val_losses)
                    if mean_val_loss < best_val_loss:
                        best_val_loss = mean_val_loss
                        best_hyperparams = (input_size, hidden_size, num_layers, dropout_size)

    # Train the best model on the full training set and evaluate it on the test set
    early_stopping = pl.callbacks.EarlyStopping(patience=20, monitor='val_loss')
    trainer = pl.Trainer(max_epochs=num_epochs, accelerator='gpu', devices=1, callbacks=[early_stopping])
    trainer.fit(best_model, train_loader, val_loader)

    # Evaluate the best model on the test set
    y_test = torch.from_numpy(y_test).float()
    y_test = y_test.view(-1, 1)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).to(device),
                                torch.tensor(y_test, dtype=torch.float32).to(device))
    test_data_loader = DataLoader(test_dataset, batch_size=96, shuffle=False)
    test_loss = trainer.test(best_model, test_data_loader)[0]['test_loss']


    print(
        f"Best hyperparameters: input_size={best_hyperparams[0]}, hidden_size={best_hyperparams[1]}, num_layers={best_hyperparams[2]}, dropout={best_hyperparams[3]}")
    print(f"Validation loss: {best_val_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")

    # Save the best model with the best hyperparameters
    torch.save(best_model.state_dict(), f"best_model{best_hyperparams[0]}_{best_hyperparams[1]}_{best_hyperparams[2]}_{best_hyperparams[3]}.pt")