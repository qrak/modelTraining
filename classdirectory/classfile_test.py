import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl


class LSTMNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size=32, num_layers=2, output_size=1, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=1)
        self.fc1_linear = nn.Linear(hidden_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2_linear = nn.Linear(64, output_size)

    @property
    def loss(self):
        return nn.L1Loss()

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.dropout(output)
        attn_output, _ = self.attn(output, output, output)
        output = output + attn_output
        if len(output.shape) == 2:
            output = output.unsqueeze(1)
        output = output[:, -1, :]
        output = self.fc1_linear(output)
        output = nn.ReLU()(self.bn1(output))
        output = self.fc2_linear(output)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.view(-1, 1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.view(-1, 1))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.view(-1, 1))
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

