import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class BitcoinPredictor(pl.LightningModule):
        def __init__(self, input_size, hidden_size=32, num_layers=2, output_size=1, dropout=0.2):
            super().__init__()

            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                                dropout=dropout)
            self.dropout = nn.Dropout(dropout)
            self.fc1_linear = nn.Linear(hidden_size, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.fc2_linear = nn.Linear(64, 16)
            self.fc3_linear = nn.Linear(16, 16)
            self.attn = nn.MultiheadAttention(16, num_heads=4)
            self.fc4_linear = nn.Linear(16, 1)
            self.loss = nn.MSELoss()
            self.weight_decay = 1e-3

        def forward(self, x):
            output, _ = self.lstm(x)
            output = self.dropout(output)
            if len(output.shape) == 2:
                output = output.unsqueeze(1)
            output = output[:, -1, :]
            output = self.fc1_linear(output)
            output = self.bn1(output)
            output = nn.functional.relu(output)
            output = output.view(output.size(0), -1)
            output = self.fc2_linear(output)
            output = nn.functional.relu(output)
            output = self.fc3_linear(output)
            output = output.unsqueeze(0)
            output = output.permute(1, 0,
                                    2)  # change from [batch, seq_len, hidden_size] to [seq_len, batch, hidden_size]
            output, _ = self.attn(output, output, output)
            output = output.squeeze(0)
            output = self.fc4_linear(output)
            output = output.view(-1)  # add this line to reshape output to have shape [batch_size]
            return output

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss(y_hat, y.view(-1))
            self.log('train_loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss(y_hat, y.view(-1))
            self.log('val_loss', loss)
            return {'val_loss': loss}

        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss(y_hat, y.view(-1))
            self.log('test_loss', loss)
            return loss

        def configure_optimizers(self):
            optimizer = optim.Adagrad(self.parameters(), lr=1e-3, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20,
                                                             verbose=True)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

        def prepare_data(self):
            # Load the CSV file into a Pandas dataframe
            df = pd.read_csv('BTC_USDT_1h_with_indicators.csv')

            # Convert the date column to a datetime object
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Set the date column as the index
            df.set_index('timestamp', inplace=True)

            # Normalize the data using the min-max scaler
            self.scaler = MinMaxScaler()

            df_norm = pd.DataFrame(self.scaler.fit_transform(df), columns=df.columns, index=df.index)

            # Convert the dataframe to PyTorch tensors
            X = torch.Tensor(df_norm.drop(columns=['close']).values)
            y = torch.Tensor(df_norm['close'].values).reshape(-1, 1)

            # Split the data into training, validation, and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

            # Move the datasets to the device

            self.train_set = TensorDataset(X_train.to(device), y_train.to(device))
            self.val_set = TensorDataset(X_val.to(device), y_val.to(device))
            self.test_set = TensorDataset(X_test.to(device), y_test.to(device))

            # Create PyTorch datasets
            self.train_set = TensorDataset(X_train, y_train)
            self.val_set = TensorDataset(X_val, y_val)
            self.test_set = TensorDataset(X_test, y_test)

        def setup(self, stage=None):
            self.prepare_data()

        def __dataloader(self, train):
            if train:
                loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=8)
            else:
                loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)
            return loader

        def train_dataloader(self):
            return self.__dataloader(train=True)

        def val_dataloader(self):
            return self.__dataloader(train=False)

        def test_dataloader(self):
            return self.__dataloader(train=False)

    # Instantiate the model
    model = BitcoinPredictor(input_size=29, hidden_size=64, num_layers=3, output_size=1, dropout=0.2)
    model.batch_size = 32
    model.to(device)

    # Define the ModelCheckpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min')

    # Train the model
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else None, max_epochs=250, callbacks=[checkpoint_callback])
    trainer.fit(model)

    # Test the model
    trainer.test(ckpt_path='best')

    # Make predictions on new data
    new_data = pd.read_csv('BTC_USDT_1h_with_indicators.csv')

    new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')

    new_data.set_index('timestamp', inplace=True)

    new_data_norm = pd.DataFrame(model.scaler.transform(new_data), columns=new_data.columns, index=new_data.index)

    X_new = torch.Tensor(new_data_norm.drop(columns=['close']).values)

    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            X_new = X_new.to(model.device)
        y_new_pred = model(X_new)
    model.train()

    y_new_pred = model.scaler.inverse_transform(y_new_pred.cpu().numpy())

