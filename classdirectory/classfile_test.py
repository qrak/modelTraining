import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F


class LSTMRegressor(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate=1e-3, weight_decay=0.0,
                 dropout=0.0, max_norm=1.0, window_size=24):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.conv1d = nn.Conv1d(in_channels=hidden_size * 2, out_channels=hidden_size, kernel_size=3, padding=1)
        self.dense = nn.Linear(hidden_size * 2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.L1Loss()
        self.max_norm = max_norm
        self.window_size = window_size

    def forward(self, x):
        # add a new dimension at position 1
        x = x.unsqueeze(1)
        # transpose to [sequence_length, batch_size, num_features]
        x = x.permute(1, 0, 2)
        # apply LSTM layer to input sequence
        lstm_out, (h_n, c_n) = self.lstm(x)
        # swap first and second dimensions
        lstm_out = lstm_out.transpose(0, 1)
        # remove singleton dimension
        lstm_out = lstm_out.squeeze(0)
        # swap second and third dimensions
        lstm_out = lstm_out.transpose(1, 2)
        # apply 1D convolution to LSTM output
        conv_out = self.conv1d(lstm_out)
        # apply ReLU activation function
        conv_out = F.relu(conv_out)
        # apply dropout regularization
        conv_out = self.dropout(conv_out)
        # swap second and third dimensions
        conv_out = conv_out.transpose(2, 0)
        # apply batch normalization to the second dimension
        conv_out = self.bn1(conv_out.transpose(1, 2))
        # apply dropout regularization
        lstm_out = self.dropout(conv_out)
        # apply fully connected layer to the output of the LSTM layer
        return self.fc(lstm_out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(2).transpose(0, 1)  # add singleton dimension and transpose
        # split x and y into chunks of window_size rows
        x_chunks = torch.split(x, self.window_size, dim=0)
        y_chunks = torch.split(y, self.window_size, dim=0)
        # iterate over the chunks and compute the loss for each one
        total_loss = 0
        for i, (x_chunk, y_chunk) in enumerate(zip(x_chunks, y_chunks)):
            y_hat_chunk = self(x_chunk)
            mse_loss = nn.MSELoss()(y_hat_chunk, y_chunk)
            l1_regularization = self.l1(self.fc.weight, torch.zeros_like(self.fc.weight))
            loss = mse_loss + self.weight_decay * l1_regularization
            total_loss += loss
            # log the loss for the first chunk for each epoch
            if i == 0:
                self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        # apply gradient normalization
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
        return {'loss': total_loss}

    def _get_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(0)  # add singleton dimension at position 0
        # split x and y into chunks of window_size rows
        x_chunks = torch.split(x, self.window_size, dim=0)
        y_chunks = torch.split(y, self.window_size, dim=0)
        # iterate over the chunks and compute the loss for each one
        total_loss = 0
        for i, (x_chunk, y_chunk) in enumerate(zip(x_chunks, y_chunks)):
            y_hat_chunk = self(x_chunk)
            mse_loss = nn.MSELoss()(y_hat_chunk, y_chunk)
            l1_regularization = self.weight_decay * torch.norm(self.fc.weight, p=1)
            loss = mse_loss + l1_regularization
            total_loss += loss
            # log the loss for the first chunk for each epoch
            if i == 0:
                self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {'loss': total_loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch', avg_loss, prog_bar=True)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss_epoch', avg_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def on_fit_start(self):
        # log hyperparameters to tensorboard
        self.logger.log_hyperparams(self.hparams)