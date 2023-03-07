import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.tensorboard import SummaryWriter

class LSTMRegressor(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate=1e-3, weight_decay=0.0,
                 dropout=0.0, max_norm=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(hidden_size * 2, hidden_size)  # add a dense layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.current_weight_decay = weight_decay
        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.L1Loss()
        self.max_norm = max_norm


    def forward(self, x):
        print(x.shape)
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

        # apply gradient normalization
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
        # Get the current learning rate and log it
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('learning_rate', lr, prog_bar=True, on_epoch=True)
        self.current_weight_decay = self.trainer.optimizers[0].param_groups[0]['weight_decay']
        self.log('weight_decay', self.current_weight_decay, prog_bar=True, on_epoch=True)
        self.log('max_norm', self._get_norm(), prog_bar=True, on_epoch=True)  # log gradient norm
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
        mse_loss = nn.MSELoss()(y_hat, y.view(-1, 1))
        l1_regularization = self.weight_decay * torch.norm(self.fc.weight, p=1)
        loss = mse_loss + l1_regularization
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {'loss': loss, 'y_hat': y_hat, 'y': y}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch', avg_loss, prog_bar=True)

    def validation_epoch_end(self, outputs):
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

