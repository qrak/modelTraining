import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

if __name__ == '__main__':
    import torch.nn.functional as F


    class LSTMNet(pl.LightningModule):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc1_linear = nn.Linear(hidden_size, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.fc2_linear = nn.Linear(64, output_size)
            self.loss = nn.MSELoss()
            self.l2_reg = nn.Linear(hidden_size, hidden_size, bias=False)
            self.l2_lambda = 1e-5

        def forward(self, x):
            # add batch dimension if necessary
            if len(x.shape) == 2:
                x = x.unsqueeze(0)

            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            output, _ = self.lstm(x, (h0, c0))
            output = self.dropout(output)
            output = self.fc1_linear(output)
            output = self.bn1(output)
            output = F.relu(output)
            output = self.fc2_linear(output)
            return output

        def training_step(self, batch, batch_idx):
            inputs, labels = batch
            outputs = self(inputs)
            l2_loss = 0
            for param in self.lstm.parameters():
                l2_loss += self.l2_lambda * torch.sum(torch.square(param))
            for param in self.l2_reg.parameters():
                l2_loss += self.l2_lambda * torch.sum(torch.square(param))
            loss = self.loss(outputs, labels.view(-1, 1)) + l2_loss
            self.log('train_loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            inputs, labels = batch
            outputs = self(inputs)
            loss = self.loss(outputs, labels.view(-1, 1))
            self.log('val_loss', loss)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

        def test_step(self, batch, batch_idx):
            inputs, labels = batch
            outputs = self(inputs)
            loss = self.loss(outputs, labels.view(-1, 1))
            self.log('test_loss', loss)


    # Load data into dataframe
    df = pd.read_csv('BTC_USDT_5m_with_indicators2.csv')
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how='any')

    scaler = StandardScaler()
    # Extract the target variable (close price)
    X = df.drop(['close', 'timestamp'], axis=1).values
    y = df['close'].values
    X = scaler.fit_transform(X)
    # Perform feature selection
    k = 10 # Number of features to select
    selector = SelectKBest(f_regression, k=k)
    X = selector.fit_transform(X, y)
    # Split the data into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, shuffle=False)
    # Move the data to the specified device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    # Define the number of folds for cross-validation
    num_folds = 4

    # Define the hyperparameters to search over
    input_sizes = [X.shape[1]]
    hidden_sizes = [16, 32, 64]
    num_layers_list = [2, 4]
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
                    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
                        print(f"Fold {fold + 1}/{num_folds}")
                        # Split the data into training and validation sets for this fold
                        X_fold_train, y_fold_train = X_train_val[train_idx], y_train_val[train_idx]
                        X_fold_val, y_fold_val = X_train_val[val_idx], y_train_val[val_idx]

                        # Create the PyTorch datasets and data loaders
                        train_dataset = TensorDataset(torch.tensor(X_fold_train, dtype=torch.float32).to(device),
                                                      torch.tensor(y_fold_train, dtype=torch.float32).to(device))
                        val_dataset = TensorDataset(torch.tensor(X_fold_val, dtype=torch.float32).to(device),
                                                    torch.tensor(y_fold_val, dtype=torch.float32).to(device))
                        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4,
                                                  pin_memory=False)
                        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4,
                                                pin_memory=False)

                        model = LSTMNet(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                        output_size=1, dropout=dropout_size).to(device)

                        # Initialize the EarlyStopping callback
                        early_stopping = pl.callbacks.EarlyStopping(patience=10, monitor='val_loss')

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
    early_stopping = pl.callbacks.EarlyStopping(patience=10, monitor='val_loss')
    trainer = pl.Trainer(max_epochs=num_epochs, accelerator='gpu', devices=1, callbacks=[early_stopping])
    trainer.fit(best_model, train_loader, val_loader)

    # Evaluate the best model on the test set
    y_test = torch.from_numpy(y_test).float()
    y_test = y_test.view(-1, 1)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)
    test_loss = trainer.test(best_model, test_loader)[0]['test_loss']


    print(
        f"Best hyperparameters: input_size={best_hyperparams[0]}, hidden_size={best_hyperparams[1]}, num_layers={best_hyperparams[2]}, dropout={best_hyperparams[3]}")
    print(f"Validation loss: {best_val_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")

    # Save the best model with the best hyperparameters
    torch.save(best_model.state_dict(), f"best_model{best_hyperparams[0]}_{best_hyperparams[1]}_{best_hyperparams[2]}_{best_hyperparams[3]}.pt")