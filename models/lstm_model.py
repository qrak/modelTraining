import torch
from torch import nn
from torch.nn.functional import softmax
from pytorch_lightning import LightningModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        self.W_query = nn.Linear(input_dim, output_dim)
        self.W_key = nn.Linear(input_dim, output_dim)
        self.W_value = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        attention_weights = softmax(torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5), dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        return attention_output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention_heads = nn.ModuleList([SelfAttention(input_dim, output_dim) for _ in range(num_heads)])

    def forward(self, x):
        attention_outputs = [head(x) for head in self.attention_heads]
        return torch.cat(attention_outputs, dim=-1)


class BitcoinPredictor(LightningModule):
    def __init__(self, config, save_dir="save"):
        super(BitcoinPredictor, self).__init__()
        self.config = config
        self.save_dir = save_dir
        self.lstm = nn.LSTM(self.config['input_size'], self.config['hidden_size'], self.config['num_layers'], batch_first=True,
                            bidirectional=True, dropout=self.config['dropout'])
        self.multi_head_attention = MultiHeadSelfAttention(2 * self.config['hidden_size'], self.config['hidden_size'],
                                                           self.config['num_heads'])
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(self.config['hidden_size'] * self.config['num_heads'], self.config['output_size'])

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        attention_output = self.multi_head_attention(lstm_output)

        # Residual connection
        residual_output = lstm_output + attention_output

        x = self.dropout(residual_output)
        x = self.linear(x[:, -1])
        return x

    def mean_absolute_percentage_error(self, y_true, y_pred):
        epsilon = 1e-8  # A small constant to avoid division by zero
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    def calculate_loss(self, y_hat, y):
        mse_loss = nn.functional.mse_loss(y_hat, y)
        return mse_loss

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y_hat, y)
        mape = self.mean_absolute_percentage_error(y, y_hat)
        return loss, mape

    def training_step(self, batch, batch_idx):
        loss, mape = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss)
        self.log('train_mape', mape)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mape = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mape', mape, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, mape = self._shared_step(batch, batch_idx)
        self.log('test_loss', loss)
        self.log('test_mape', mape)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'],
                                     weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
