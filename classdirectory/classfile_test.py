import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F

input_size = 29
hidden_size = 32
num_layers = 8
output_size = 1
class LSTMRegressor(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate=1e-3, weight_decay=0.0,
                 dropout=0.0, max_norm=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.dense = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.L1Loss()
        self.max_norm = max_norm
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = lstm_out.unsqueeze(2)
        lstm_out = lstm_out.permute(0, 2, 1)
        conv_out = self.conv1d(lstm_out)
        conv_out = self.conv1d_2(F.relu(conv_out))
        conv_out = self.conv1d_3(F.relu(conv_out))
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.transpose(1, 2)
        lstm_out = self.bn2(F.relu(conv_out))
        lstm_out = self.dropout(lstm_out)
        dense_out = F.relu(self.dense(lstm_out.flatten(start_dim=1)))
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

        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_weight_decay', self.weight_decay, prog_bar=True, on_epoch=True)
        self.log('train_learning_rate', self.learning_rate, prog_bar=True, on_epoch=True)
        self.log('max_norm', self._get_norm(), prog_bar=True)  # log gradient norm
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
        self.log('val_weight_decay', self.weight_decay, prog_bar=True, on_epoch=True)
        self.log('val_learning_rate', self.learning_rate, prog_bar=True, on_epoch=True)
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