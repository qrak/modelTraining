import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from os import path
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def create_input_target_tensors(df, seq_len, target_col, timestamp_col):
    df = df.drop(columns=[timestamp_col])  # Drop the 'timestamp' column

    data = df.to_numpy()

    input_data, target_data = [], []
    for i in range(len(data) - seq_len):
        input_data.append(np.delete(data[i:i + seq_len], df.columns.get_loc(target_col), axis=1))
        target_data.append(data[i + seq_len][df.columns.get_loc(target_col)])

    input_data = np.array(input_data)
    target_data = np.array(target_data)

    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    target_tensor = torch.tensor(target_data, dtype=torch.float32)

    return input_tensor, target_tensor


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

        attention_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5), dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        return attention_output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention_heads = nn.ModuleList([SelfAttention(input_dim, output_dim) for _ in range(num_heads)])

    def forward(self, x):
        attention_outputs = [head(x) for head in self.attention_heads]
        return torch.cat(attention_outputs, dim=-1)


class BitcoinPredictor(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers, sequence_length, num_heads):
        super(BitcoinPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        # self.gru = nn.GRU(2 * hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2)  # Add a GRU layer
        self.multi_head_attention = MultiHeadSelfAttention(2 * hidden_size, hidden_size, num_heads)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size * num_heads, output_size)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        attention_output = self.multi_head_attention(lstm_output)

        # Residual connection
        residual_output = lstm_output + attention_output

        x = self.dropout(residual_output)
        x = self.linear(x[:, -1])
        return x.squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, min_lr=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }


def normalize_data(df, target_column='close'):
    # Store the timestamp and target_column separately
    timestamp = df['timestamp']
    target = df[target_column]

    # Drop the timestamp and target_column from the DataFrame
    df = df.drop(columns=['timestamp', target_column])

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()  # Add a separate scaler for the target column
    normalized_data = scaler.fit_transform(df)
    target_normalized = target_scaler.fit_transform(
        target.to_numpy().reshape(-1, 1))  # Normalize the target column separately

    # Convert the normalized data back to a DataFrame
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

    # Add the timestamp and target_column back to the normalized DataFrame
    normalized_df['timestamp'] = timestamp
    normalized_df[target_column] = target_normalized.flatten()  # Add the normalized target column

    return normalized_df, scaler, target_scaler


def evaluate_and_plot(model, val_loader, target_scaler):
    model.eval()
    all_actual_values = []
    all_predicted_values = []

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            y_pred = model(x)

            all_actual_values.extend(y.numpy())
            all_predicted_values.extend(y_pred.numpy())

    # Inverse transform the predicted and actual values
    all_actual_values = target_scaler.inverse_transform(np.array(all_actual_values).reshape(-1, 1))
    all_predicted_values = target_scaler.inverse_transform(np.array(all_predicted_values).reshape(-1, 1))

    plot_predictions(all_actual_values, all_predicted_values)


def plot_predictions(actual, predicted):
    plt.figure(figsize=(14, 8))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predicted, label='Predicted', color='orange')
    plt.xlabel('Time steps')
    plt.ylabel('Bitcoin Price')
    plt.legend()
    plt.show()


def main():
    # Load and normalize your historical OHLCV data and indicators here
    df = pd.read_csv('csv_modified/BTC_USDT_1h_indicators4.csv')
    # df = df[['timestamp', 'open', 'high', 'low', 'close']]

    train_val_df, test_df = train_test_split(df, test_size=0.1, shuffle=False)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, shuffle=False)
    train_val_df_normalized, _, target_scaler = normalize_data(train_val_df)
    train_df_normalized, _, _ = normalize_data(train_df)
    val_df_normalized, _, _ = normalize_data(val_df)
    test_df_normalized, _, _ = normalize_data(test_df)

    sequence_length = 24
    batch_size = 64

    train_input, train_target = create_input_target_tensors(train_df_normalized, sequence_length, 'close', 'timestamp')
    val_input, val_target = create_input_target_tensors(val_df_normalized, sequence_length, 'close', 'timestamp')
    test_input, test_target = create_input_target_tensors(test_df_normalized, sequence_length, 'close', 'timestamp')

    # Create the TensorDatasets
    train_dataset = TensorDataset(train_input, train_target)
    val_dataset = TensorDataset(val_input, val_target)
    test_dataset = TensorDataset(test_input, test_target)  # Create the TensorDataset for test set

    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             drop_last=True)  # Create the DataLoader for test set

    input_size = train_df.shape[1] - 2
    hidden_size = 128
    output_size = 1
    num_layers = 2
    num_heads = 2

    model = BitcoinPredictor(input_size, hidden_size, output_size, num_layers, sequence_length, num_heads)
    logger = TensorBoardLogger('tb_logs', name='bitcoin_predictor')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath='checkpoints/',
        filename='best_model_{input_size}_{hidden_size}',
        save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        log_every_n_steps=1,
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    # Load the last best model
    # best_model_path = checkpoint_callback.best_model_path
    # model = BitcoinPredictor.load_from_checkpoint(best_model_path)

    trainer.fit(model, train_loader, val_loader)
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    dir_path = "save"
    file_name = f"best_model_{input_size}_{hidden_size}_{num_layers}_{num_heads}_{time_stamp}.pt"
    file_path = path.join(dir_path, file_name)
    torch.save(model.state_dict(), file_path)

    trainer.test(model, test_loader)  # Evaluate on the test set
    # Evaluate and plot predictions
    evaluate_and_plot(model, test_loader, target_scaler)  # Pass the target_scaler


if __name__ == '__main__':
    main()
