import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    class LSTMRegressor(pl.LightningModule):
        def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate=1e-3, weight_decay=0.0,
                     dropout=0.0):
            super().__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.dense = nn.Linear(hidden_size * 2, hidden_size)  # add a dense layer
            self.fc = nn.Linear(hidden_size, output_size)
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.dropout = nn.Dropout(dropout)
            self.l1 = nn.L1Loss()

        def forward(self, x):
            lstm_out, (h_n, c_n) = self.lstm(x)
            h_n = h_n[-1]  # use last hidden state from final layer
            dense_out = F.relu(self.dense(lstm_out))
            dense_out = self.dropout(dense_out)
            output = self.fc(dense_out)
            return output

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            l1_regularization = self.l1(self.fc.weight, torch.zeros_like(self.fc.weight))
            mse_loss = nn.MSELoss()(y_hat, y.view(-1, 1))
            loss = mse_loss + self.weight_decay * l1_regularization
            self.log('train_loss', loss, prog_bar=True)
            return {'loss': loss, 'y_hat': y_hat, 'y': y}

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            mse_loss = nn.MSELoss()(y_hat, y.view(-1, 1))
            l1_regularization = self.weight_decay * torch.norm(self.fc.weight, p=1)
            loss = mse_loss + l1_regularization
            self.log('val_loss', loss, prog_bar=True)
            return {'loss': loss, 'y_hat': y_hat, 'y': y}

        def training_epoch_end(self, outputs):
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
            self.log('train_loss_epoch', avg_loss, prog_bar=True)

        def validation_epoch_end(self, outputs):
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
            self.log('val_loss_epoch', avg_loss, prog_bar=True)

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'
            }

        def on_fit_start(self):
            # log hyperparameters to tensorboard
            self.logger.log_hyperparams(self.hparams)

    # load data
    df = pd.read_csv("BTC_USDT_1h_with_indicators.csv")
    # Convert the date column to a datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Set the date column as the index
    df.set_index('timestamp', inplace=True)

    features = df.drop(['close'], axis=1).values
    labels = df['close'].values.reshape(-1, 1)

    # scale data
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    labels_scaled = scaler.fit_transform(labels)

    # split dataset into train, val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(features_scaled, labels_scaled, test_size=0.1,
                                                                random_state=42, shuffle=False)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42,
                                                      shuffle=False)

    # convert to tensors
    features_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    labels_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    features_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    labels_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    features_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    labels_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # create datasets
    train_dataset = TensorDataset(features_train_tensor, labels_train_tensor)
    val_dataset = TensorDataset(features_val_tensor, labels_val_tensor)
    test_dataset = TensorDataset(features_test_tensor, labels_test_tensor)

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, drop_last=True)

    # create model
    input_size = features.shape[1]
    hidden_size = 32
    num_layers = 12
    output_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-3
    weight_decay = 1e-4
    dropout = 0.2

    model = LSTMRegressor(input_size, hidden_size, num_layers, output_size,
                          learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout).to(device)

    # create checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best_model_{epoch}_{val_loss:.4f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=15, mode="min")
    # create hyperparameters dictionary
    # initialize logger
    logger = TensorBoardLogger(save_dir='./lightning_logs', name='bilstm-regressor')
    # train model
    trainer = pl.Trainer(max_epochs=200, accelerator="gpu" if torch.cuda.is_available() else 0,
                         logger=logger,
                         callbacks=[checkpoint_callback, early_stopping_callback])
    trainer.fit(model, train_loader, val_loader)

    # switch to evaluation mode
    model.eval()
    model.to(device)

    # make predictions on test set
    test_pred = []
    test_actual = []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            y_pred = model(x)
            test_pred.append(y_pred.cpu().numpy())
            test_actual.append(y.cpu().numpy())

    # concatenate all batches
    test_pred = np.concatenate(test_pred, axis=0)
    test_actual = np.concatenate(test_actual, axis=0)

    # unscale predictions and actual values
    test_pred_unscaled = scaler.inverse_transform(test_pred.reshape(-1, 1))
    test_actual_unscaled = scaler.inverse_transform(test_actual.reshape(-1, 1))

    # print predicted and actual values

    import matplotlib.pyplot as plt

    # plot predicted and actual values
    test_df = df.tail(len(test_actual_unscaled))

    plt.plot(test_df.index, test_actual_unscaled, label='actual')
    plt.plot(test_df.index, test_pred_unscaled, label='predicted')
    plt.legend()
    plt.show()