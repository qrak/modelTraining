import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, TensorDataset
from classdirectory.classfile import LSTMNet
#import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    data = pd.read_csv('BTC_USDT_1h_with_indicators.csv', parse_dates=['timestamp'])
    # Split the data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        data.drop(['timestamp', 'close'], axis=1),
        data['close'],
        test_size=0.2,
        shuffle=False,
        random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        shuffle=False,
        random_state=42
    )

    # Fit the StandardScaler on the training set only and apply it to the validation and test sets
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Scale the target variable
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).squeeze()
    y_val = y_scaler.transform(y_val.values.reshape(-1, 1)).squeeze()
    y_test = y_scaler.transform(y_test.values.reshape(-1, 1)).squeeze()

    # Define the hyperparameters to search over
    input_size = (X_train.shape[1])
    hidden_size = 64
    num_layers = 16
    dropout_size = 0.1
    #torch.set_float32_matmul_precision('high')
    num_epochs = 200

    print(
        f"Training model with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout_size}")

    # Create the PyTorch datasets and data loaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                                  torch.tensor(y_train, dtype=torch.float32).to(device))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).to(device),
                                torch.tensor(y_val, dtype=torch.float32).to(device))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).to(device),
                                 torch.tensor(y_test, dtype=torch.float32).to(device))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = LSTMNet(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                    output_size=1, dropout=dropout_size).to(device)

    # Initialize the EarlyStopping callback and the ModelCheckpoint callback
    early_stopping = pl.callbacks.EarlyStopping(patience=50, monitor='val_loss')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best_model_{epoch}_{val_loss:.4f}',
        save_top_k=1,
        mode='min'
    )

    # Train the model
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else None,
        callbacks=[early_stopping, checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model on the validation set for this fold
    val_loss = trainer.validate(model, val_loader)[0]['val_loss']
    mean_val_loss = np.mean(val_loss)
    print(f"Mean Validation Loss: {mean_val_loss:.6f}")

    # Save the best model
    best_model_path = f"save/best_model_{input_size}_{hidden_size}_{num_layers}_{dropout_size}.pt"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    torch.save(model.state_dict(), best_model_path)

    # Evaluate the best model on the test set
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).to(device),
                                 torch.tensor(y_test, dtype=torch.float32).to(device))
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # Load the best model
    best_model = LSTMNet(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=1,
                         dropout=dropout_size)
    best_model.to(device)
    best_model.load_state_dict(torch.load(best_model_path))



    test_loss = trainer.test(best_model, test_data_loader)[0]['test_loss']
    print(f"Test loss: {test_loss:.6f}")

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
