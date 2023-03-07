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
hidden_size = 64
num_layers = 2
output_size = 1

learning_rate = 1e-3
weight_decay = 1e-4
dropout = 0.2

"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMRegressor(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate=1e-3, weight_decay=1e-5,
                 dropout=0.0, max_norm=0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=hidden_size*2, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.dense = nn.Linear(hidden_size * 2, hidden_size)  # add a dense layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.current_weight_decay = weight_decay
        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.L1Loss()
        self.max_norm = max_norm

    def forward(self, x):
        x = x.permute(0, 2, 1)  # permute input for Conv1d layer
        x = self.conv1d(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)  # permute input back to original shape for LSTM layer
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # use only the last hidden state
        x = self.dense(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        l1_regularization = self.weight_decay * torch.norm(self.fc.weight, p=1)  # change to L1 regularization
        mse_loss = F.mse_loss(y_hat, y)
        loss = mse_loss + l1_regularization

        # apply gradient normalization
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)

        # Get the current learning rate and log it
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('learning_rate', lr, prog_bar=True, on_epoch=True)

        # Log weight decay and gradient norm
        weight_decay = self.trainer.optimizers[0].param_groups[0]['weight_decay']
        self.log('weight_decay', weight_decay, prog_bar=True, on_epoch=True)
        gradient_norm = self._get_norm()
        self.log('max_norm', gradient_norm, prog_bar=True, on_epoch=True)

        return {'loss': loss, 'y_hat': y_hat, 'y': y}

    def _get_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse_loss = F.mse_loss(y_hat, y)
        l1_regularization = self.weight_decay * torch.norm(self.fc.weight, p=1)
        loss = mse_loss + l1_regularization
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {'loss': loss, 'y_hat': y_hat, 'y': y}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch', avg_loss, prog_bar=True)

    def validation_epoch_end(self, outputs):
        pass
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss_epoch', avg_loss, prog_bar=True)
        return {'val_loss': avg_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('test_loss_epoch', avg_loss, prog_bar=True)
        return {'test_loss': avg_loss}


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        weight_decay_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + epoch))
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'weight_decay_scheduler': weight_decay_scheduler,
            'monitor': 'val_loss'  # Optional: specify the metric to monitor
        }


    def on_fit_start(self):
        # log hyperparameters to tensorboard
        self.logger.log_hyperparams(self.hparams)

