import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

from torch.utils.data import DataLoader, TensorDataset
from classdirectory.classfile_test import LSTMNet
#import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    data = pd.read_csv('BTC_USDT_1h_with_indicators.csv', parse_dates=['timestamp'])
    features = data.drop(['timestamp', 'close'], axis=1)
    target = data['close'].values.reshape(-1, 1)
    # Initialize the scaler
    scaler = MinMaxScaler()
    scaler_target = MinMaxScaler()
    # Fit the scaler to the features
    scaler.fit(features)
    scaler_target.fit(target)
    # Scale the features
    scaled_features = scaler.transform(features)
    scaled_target = scaler_target.transform(target)
      # Scale the features
    # Split the data into training and testing sets
    tscv = TimeSeriesSplit(n_splits=5)


    # Define the hyperparameters to search over
    input_size = (29)
    hidden_size = 32
    num_layers = 8
    dropout_size = 0.2
    #torch.set_float32_matmul_precision('high')
    num_epochs = 200

    print(
        f"Training model with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout_size}")
    for train_val_index, test_index in tscv.split(scaled_features):
        X_train_val, X_test = scaled_features[train_val_index], scaled_features[test_index]
        y_train_val, y_test = scaled_target[train_val_index], scaled_target[test_index]
        split_idx = int(len(X_train_val) * 0.8)
        X_train, X_val = X_train_val[:split_idx], X_train_val[split_idx:]
        y_train, y_val = y_train_val[:split_idx], y_train_val[split_idx:]
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
