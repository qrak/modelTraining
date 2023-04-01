import torch
from torch import nn
from torch.nn import LSTM, LayerNorm, Module, Linear, BatchNorm1d, Conv1d
from torch.nn.functional import softmax
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler, RMSprop
from pytorch_lightning import LightningModule
from torch.nn.init import xavier_uniform_
from torch.nn import ReLU

from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        self.W_query = nn.Linear(input_dim, output_dim)
        self.W_key = nn.Linear(input_dim, output_dim)
        self.W_value = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        attention_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5), dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        return attention_output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention_heads = nn.ModuleList([SelfAttention(input_dim, output_dim) for _ in range(num_heads)])

    def forward(self, x):
        attention_outputs = [head(x) for head in self.attention_heads]
        return torch.cat(attention_outputs, dim=-1)


class BitcoinPredictor(LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, sequence_length, batch_size,
                 num_epochs, learning_rate, weight_decay, dropout, save_dir="save"):
        super(BitcoinPredictor, self).__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.input_size = input_size
        self.save_dir = save_dir
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=self.dropout)
        self.multi_head_attention = MultiHeadSelfAttention(2 * hidden_size, hidden_size, num_heads)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size * num_heads, output_size)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        attention_output = self.multi_head_attention(lstm_output)

        # Residual connection
        residual_output = lstm_output + attention_output

        x = self.dropout(residual_output)
        x = self.linear(x[:, -1])
        return x

    def mean_absolute_percentage_error(self, y_true, y_pred):
        epsilon = 1e-8  # A small constant to avoid division by zero
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    """def calculate_loss(self, y_hat, y):
        mae_loss = F.l1_loss(y_hat, y)
        loss = mae_loss
        return loss"""
    def calculate_loss(self, y_hat, y):
        mse_loss = F.mse_loss(y_hat, y)
        loss = mse_loss
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y_hat, y)
        mape = self.mean_absolute_percentage_error(y, y_hat)
        self.log('train_loss', loss)
        self.log('train_mape', mape)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y_hat, y)
        mape = self.mean_absolute_percentage_error(y, y_hat)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mape', mape, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y_hat, y)
        mape = self.mean_absolute_percentage_error(y, y_hat)
        self.log('test_loss', loss)
        self.log('test_mape', mape)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
