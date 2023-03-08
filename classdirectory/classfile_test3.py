import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import math


from torch.utils.tensorboard import SummaryWriter

"""
COMMENTED FOR INFORMATION WHAT I FEED TO THE MODEL
batch_size = 64
input_size = 18
hidden_size = 64
num_layers = 2
output_size = 1
output_size = 1

learning_rate = 1e-3
weight_decay = 1e-4
dropout = 0.2

"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMRegressor(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, sequence_length=24, learning_rate=1e-1,
                 weight_decay=1e-5,
                 dropout=0.0, max_norm=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_norm = max_norm
        self.weight_decay_scheduler = WeightDecayScheduler(weight_decay)
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size * 2, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm([hidden_size * 4, sequence_length // 3])
        self.fc1 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.L1Loss()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, hidden_size*4)

        # apply layer normalization and feedforward layers
        x = self.layer_norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Add L1 and L2 regularization terms
        l1_regularization = self.weight_decay * (torch.norm(self.fc1.weight, p=1) + torch.norm(self.fc2.weight, p=1))
        l2_regularization = self.weight_decay * (torch.norm(self.fc1.weight, p=2) + torch.norm(self.fc2.weight, p=2))

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
        l1_regularization = self.weight_decay * (torch.norm(self.fc1.weight, p=1) + torch.norm(self.fc2.weight, p=1))
        l2_regularization = self.weight_decay * (torch.norm(self.fc1.weight, p=2) + torch.norm(self.fc2.weight, p=2))

        mse_loss = F.mse_loss(y_hat, y)
        loss = mse_loss + l1_regularization + l2_regularization
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {'loss': loss, 'y_hat': y_hat, 'y': y}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, eps=1e-8)
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
