import torch
from torch.nn import LSTM, LayerNorm, Module, Linear
from torch.nn.functional import softmax
from torch.nn.functional import mse_loss
from torch.optim import Adam, lr_scheduler
from pytorch_lightning import LightningModule
from torch.nn.init import xavier_uniform_


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


class SelfAttention(Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.query = Linear(hidden_size, hidden_size)
        self.key = Linear(hidden_size, hidden_size)
        self.value = Linear(hidden_size, hidden_size)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        energy = torch.bmm(queries, keys.transpose(1, 2)) / self.scale
        attention = softmax(energy, dim=2)
        weighted = torch.bmm(attention, values)
        return weighted


class LSTMRegressor(LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate, dropout, weight_decay, gradient_norm=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.mse_loss = mse_loss
        self.regularization_strength_l1 = 0.001
        self.regularization_strength_l2 = 0.001
        self.lstm1 = LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout,
                          bidirectional=True)
        self.fc1 = Linear(hidden_size * 2, hidden_size * 2) # added linear layer after LSTM
        xavier_uniform_(self.fc1.weight)
        self.attention = SelfAttention(hidden_size * 2)
        self.norm1 = LayerNorm(hidden_size * 2)
        self.fc2 = Linear(hidden_size * 2, output_size) # added linear layer after self-attention
        xavier_uniform_(self.fc2.weight)


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
        mse_loss = self.mse_loss(y_hat, y)
        l1_loss = sum(p.abs().sum() for p in self.parameters()) * self.regularization_strength_l1
        l2_loss = sum(p.pow(2).sum() for p in self.parameters()) * self.regularization_strength_l2
        loss = mse_loss + l1_loss + l2_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse_loss = self.mse_loss(y_hat, y)
        l1_loss = sum(p.abs().sum() for p in self.parameters()) * self.regularization_strength_l1
        l2_loss = sum(p.pow(2).sum() for p in self.parameters()) * self.regularization_strength_l2
        loss = mse_loss + l1_loss + l2_loss
        self.log('val_loss', loss, prog_bar=True)
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse_loss = self.mse_loss(y_hat, y)
        loss = mse_loss
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True,
                                                   eps=1e-8)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',  # Optional: specify the metric to monitor
        }

