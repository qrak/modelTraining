import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

class LSTMNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size=32, num_layers=2, output_size=1, dropout=0.1):
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
        output = output.permute(1, 0, 2)  # change from [batch, seq_len, hidden_size] to [seq_len, batch, hidden_size]
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

