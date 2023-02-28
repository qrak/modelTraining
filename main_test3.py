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
    class LSTMNet(pl.LightningModule):
      def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
          super().__init__()
          self.input_size = input_size  # add this line
          self.hidden_size = hidden_size
          self.num_layers = num_layers
          self.output_size = output_size
          self.dropout = dropout
          self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
          self.dropout1 = nn.Dropout(dropout)
          self.fc1 = nn.Linear(hidden_size, 64)
          self.dropout2 = nn.Dropout(dropout)
          self.fc2 = nn.Linear(64, output_size)
          self.loss = nn.MSELoss()

      def forward(self, x):
          out, _ = self.lstm(x)
          out = self.dropout1(out)
          out = self.fc1(out[:, -1, :])
          out = self.dropout2(out)
          out = self.fc2(out)
          return out

      def training_step(self, batch, batch_idx):
          inputs, labels = batch
          inputs = inputs.unsqueeze(1)  # reshape inputs to have a sequence length of 1
          outputs = self(inputs)
          loss = self.loss(outputs, labels.view(-1, 1))
          self.log('train_loss', loss)
          return loss

      def validation_step(self, batch, batch_idx):
          inputs, labels = batch
          outputs = self(inputs.unsqueeze(1))
          loss = self.loss(outputs, labels.view(-1, 1))
          self.log('val_loss', loss.item())

      def configure_optimizers(self):
          optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
          return optimizer

      def test_step(self, batch, batch_idx):
          inputs, labels = batch
          outputs = self(inputs.unsqueeze(1))
          loss = self.loss(outputs, labels.view(-1, 1))
          self.log('test_loss', loss)
    # Load data into dataframe
    df = pd.read_csv('BTC_USDT_5m_with_indicators2.csv')
    df = df.dropna()
    # drop rows with infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    scaler = StandardScaler()
    # Extract the target variable (close price)
    X = df.drop(['close', 'timestamp'], axis=1).values
    y = df['close'].values
    X = scaler.fit_transform(X)
    # Perform feature selection
    k = 8  # Number of features to select
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
    num_folds = 2

    # Define the hyperparameters to search over
    input_sizes = [X.shape[1]]
    hidden_sizes = [32, 64]
    num_layers_list = [4]
    dropout_sizes = [0.1]

    num_epochs = 1
    best_val_loss = float('inf')
    train_loader = None
    optimizer = None
    criterion = None
    val_loader = None
    best_hyperparams = None
    best_fold = None
    patience = 10
    for input_size in input_sizes:
        for hidden_size in hidden_sizes:
            for num_layers in num_layers_list:
                for dropout_size in dropout_sizes:
                    print(f"Training model with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout_size}")
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
                        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=False)
                        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=False)
                        # Create the model, loss function, and optimizer
                        model = LSTMNet(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                        output_size=1, dropout=dropout_size).to(device)
                        optimizer = optim.Adam(model.parameters(), lr=1e-3)
                        criterion = nn.MSELoss()
                        # Initialize the EarlyStopping callback
                        early_stopping = pl.callbacks.EarlyStopping(patience=10, monitor='val_loss')

                        # Train the model
                        trainer = pl.Trainer(max_epochs=num_epochs, gpus=1, callbacks=[early_stopping])
                        trainer.fit(model, train_loader, val_loader)

                        # Evaluate the model on the validation set for this fold
                        val_loss = trainer.validate(model, val_loader)[0]['val_loss']
                        val_losses.append(val_loss)

                        mean_val_loss = np.mean(val_losses[:fold + 1])
                        print(f"Fold {fold + 1}/{num_folds}, Mean Validation Loss: {mean_val_loss:.6f}")

                        # Check if this model has a lower validation loss than the previous best model
                        if val_loss < best_fold_val_loss:
                          # Save the model
                          model_dir = f"model_fold{fold+1}_input_size={input_size}_hidden_size={hidden_size}_num_layers={num_layers}_dropout={dropout_size}"
                          os.makedirs(model_dir, exist_ok=True)
                          model_filename = os.path.join(model_dir, "best_model.pt")
                          torch.save(model.state_dict(), model_filename)
                          best_fold_val_loss = val_loss
                          best_val_losses.append(val_loss)
                          # Save the validation loss in the model directory
                          val_loss_filename = os.path.join(model_dir, "best_val_loss.txt")
                          with open(val_loss_filename, "w") as f:
                              f.write(str(val_loss))
                          # Check if this fold has a lower validation loss than the previous best fold
                          if val_loss < best_val_loss:
                              best_val_loss = val_loss
                              best_hyperparams = (input_size, hidden_size, num_layers, dropout_size)
                              # Stop early if the mean validation loss has not improved in the last `patience` folds
                          if fold >= patience - 1:
                              if mean_val_loss > np.mean(best_val_losses[-patience:]):
                                  print(
                                      f"No improvement in mean validation loss for the last {patience} folds. Stopping early.")
                                  break

    print(
        f"Best hyperparameters: input_size={best_hyperparams[0]}, hidden_size={best_hyperparams[1]}, num_layers={best_hyperparams[2]}, dropout={best_hyperparams[3]}")
    print(f"Validation loss: {best_val_loss:.6f}")

    # Load the best model with the best hyperparameters and evaluate it on the test set
    best_model_dir = f"model_fold{best_fold + 1}_input_size={best_hyperparams[0]}_hidden_size={best_hyperparams[1]}_num_layers={best_hyperparams[2]}_dropout={best_hyperparams[3]}"
    best_model_filename = os.path.join(best_model_dir, "best_model.pt")
    best_model = LSTMNet(input_size=best_hyperparams[0], hidden_size=best_hyperparams[1],
                         num_layers=best_hyperparams[2], output_size=1, dropout=best_hyperparams[3]).to(device)
    best_model.load_state_dict(torch.load(best_model_filename))
    y_test = torch.from_numpy(y_test).float().to(device)
    test_loss = criterion(best_model(X_test), y_test).item()
    print(f"Test loss: {test_loss:.6f}")

    # Save the best model with the best hyperparameters
    torch.save(best_model.state_dict(), "best_model.pt")