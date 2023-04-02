import torch
import os
from torch import nn
from pytorch_lightning import LightningModule
from models.attention_mechanisms import MultiHeadSelfAttention
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class StockCryptoPredictor(LightningModule):
    """
    A PyTorch Lightning module for predicting future values of stocks or cryptocurrencies using a multi-head self-attention LSTM network.

    Args:
        config (dict): A dictionary of configuration parameters for the model.
        save_dir (str): The directory to save the model checkpoints.

    Attributes:
        lstm (nn.LSTM): The LSTM layer of the model.
        multi_head_attention (MultiHeadSelfAttention): The multi-head self-attention layer of the model.
        dropout (nn.Dropout): The dropout layer of the model.
        linear (nn.Linear): The linear layer of the model.
    """
    def __init__(self, config, save_dir="save"):
        super(StockCryptoPredictor, self).__init__()
        self.config = config
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.lstm = nn.LSTM(self.config['input_size'], self.config['hidden_size'], self.config['num_layers'],
                            batch_first=True,
                            bidirectional=True,
                            dropout=self.config['dropout'])
        self.multi_head_attention = MultiHeadSelfAttention(2 * self.config['hidden_size'], self.config['hidden_size'],
                                                           self.config['num_heads'])
        self.dropout = nn.Dropout(self.config['dropout'])
        self.linear = nn.Linear(2 * self.config['hidden_size'], self.config['output_size'])

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        attention_output = self.multi_head_attention(lstm_output)

        # Residual connection
        residual_output = lstm_output + attention_output

        x = self.dropout(residual_output)
        x = self.linear(x[:, -1])
        return x

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        epsilon = 1e-8  # A small constant to avoid division by zero
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    def calculate_loss(self, y_hat, y):
        mse_loss = nn.functional.mse_loss(y_hat, y)
        return mse_loss

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y_hat, y)
        mape = self.mean_absolute_percentage_error(y, y_hat)
        return loss, mape

    def training_step(self, batch, batch_idx):
        loss, mape = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss)
        self.log('train_mape', mape)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mape = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_mape', mape, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, mape = self._shared_step(batch, batch_idx)
        self.log('test_loss', loss)
        self.log('test_mape', mape)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'],
                                     weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
