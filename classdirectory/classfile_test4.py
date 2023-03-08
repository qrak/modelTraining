import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import os


from torch.utils.tensorboard import SummaryWriter

"""
COMMENTED FOR INFORMATION WHAT I FEED TO THE MODEL
batch_size = 64
input_size = 32
hidden_size = 96
num_layers = 3
output_size = 1
output_size = 1

learning_rate = 1e-3
weight_decay = 1e-4
dropout = 0.2

"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import torch.nn.functional as F

class LSTMRegressor(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate=1e-3, weight_decay=0.0,
                 dropout=0.0, max_norm=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_size*2) # add layer normalization layer
        self.dense = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.current_weight_decay = weight_decay
        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.L1Loss()
        self.max_norm = max_norm

        # add self-attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=8, dropout=dropout)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_n = h_n[-1]
        dense_out = F.relu(self.dense(lstm_out))
        dense_out = dense_out.view(-1, dense_out.size(2),
                                   dense_out.size(1))  # reshape to (batch_size, hidden_size*2, sequence_length)
        dense_out = self.layer_norm(dense_out)
        dense_out = dense_out.view(-1, dense_out.size(2),
                                   dense_out.size(1))  # reshape back to (batch_size, sequence_length, hidden_size*2)
        dense_out = self.dropout(dense_out)

        # apply self-attention mechanism
        lstm_out = lstm_out.transpose(0, 1)
        lstm_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_out = lstm_out.transpose(0, 1)

        output = self.fc(lstm_out)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Add L1 and L2 regularization terms
        l1_regularization = self.weight_decay * (
                    torch.norm(self.fc1.weight, p=1) + torch.norm(self.fc2.weight, p=1))
        l2_regularization = self.weight_decay * (
                    torch.norm(self.fc1.weight, p=2) + torch.norm(self.fc2.weight, p=2))

        mse_loss = F.mse_loss(y_hat, y)
        loss = mse_loss + l1_regularization + l2_regularization

        # apply gradient normalization
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)

        # Get the current learning rate and log it
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('learning_rate', lr, prog_bar=True, on_epoch=True)
        gradient_norm = self._get_norm()
        self.log('max_norm', gradient_norm, prog_bar=True, on_epoch=True)

        return {'loss': loss, 'y_hat': y_hat, 'y': y}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        weight_decay = self.weight_decay_scheduler(self.current_epoch, loss)
        self.log('weight_decay', weight_decay, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        l1_regularization = self.weight_decay * (
                torch.norm(self.fc1.weight, p=1) + torch.norm(self.fc2.weight, p=1))
        l2_regularization = self.weight_decay * (
                torch.norm(self.fc1.weight, p=2) + torch.norm(self.fc2.weight, p=2))

        mse_loss = F.mse_loss(y_hat, y)
        loss = mse_loss + l1_regularization + l2_regularization
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {'loss': loss, 'y_hat': y_hat, 'y': y}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                         verbose=True, eps=1e-8)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'  # Optional: specify the metric to monitor
        }

    def on_fit_start(self):
        # log hyperparameters to tensorboard
        self.logger.log_hyperparams(self.hparams)

    def _get_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

class WeightDecayScheduler:
    def __init__(self, weight_decay, factor=0.5, patience=5):
        self.weight_decay = weight_decay
        self.factor = factor
        self.patience = patience
        self.num_bad_epochs = 0
        self.best_loss = float('inf')

    def __call__(self, epoch, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.weight_decay *= self.factor
                self.num_bad_epochs = 0
                print(f'Reducing weight decay to {self.weight_decay:.8f}')

        return self.weight_decay
