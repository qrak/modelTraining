import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.init as init


from torch.utils.tensorboard import SummaryWriter

"""
COMMENTED FOR INFORMATION WHAT I FEED TO THE MODEL
batch_size = 32
input_size = 32
hidden_size = 96
num_layers = 4
output_size = 1


learning_rate = 1e-3
weight_decay = 1e-4
dropout = 0.2

"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        energy = torch.bmm(queries, keys.transpose(1, 2)) / self.scale
        attention = F.softmax(energy, dim=2)
        weighted = torch.bmm(attention, values)
        return weighted


class LSTMRegressor(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate, dropout, weight_decay):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.regularization_strength_l1 = 0.001
        self.regularization_strength_l2 = 0.001
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout,
                             bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size*2) # added linear layer after LSTM
        init.xavier_uniform_(self.fc1.weight)
        self.attention = SelfAttention(hidden_size * 2)
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, output_size) # added linear layer after self-attention
        init.xavier_uniform_(self.fc2.weight)


    def forward(self, x):
        lstm1_out, _ = self.lstm1(x)
        fc_out = self.fc1(lstm1_out) # apply linear layer after LSTM
        attention_out = self.attention(fc_out) # apply self-attention on output of linear layer
        norm_out = self.norm1(attention_out)
        output = self.fc2(norm_out[:, -1, :])
        return output


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse_loss = F.mse_loss(y_hat, y)
        l1_loss = sum(p.abs().sum() for p in self.parameters()) * self.regularization_strength_l1
        l2_loss = sum(p.pow(2).sum() for p in self.parameters()) * self.regularization_strength_l2
        loss = mse_loss + l1_loss + l2_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse_loss = F.mse_loss(y_hat, y)
        l1_loss = sum(p.abs().sum() for p in self.parameters()) * self.regularization_strength_l1
        l2_loss = sum(p.pow(2).sum() for p in self.parameters()) * self.regularization_strength_l2
        loss = mse_loss + l1_loss + l2_loss
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True,
                                                         eps=1e-8)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'  # Optional: specify the metric to monitor
        }
