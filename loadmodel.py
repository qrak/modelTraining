from classdirectory.classfile import LSTMNet
import ccxt
import torch
import pandas as pd
import numpy as np
from classdirectory.classfile import LSTMNet


# Load the pre-trained model
best_model_path = 'best_model.pt'
best_model = LSTMNet(input_size=10, hidden_size=64, num_layers=2, output_size=1, dropout=0.1)
best_model.load_state_dict(torch.load(best_model_path))
scaler = StandardScaler()
# Load the test data
test_data = pd.read_csv('BTC_USDT_5m_with_indicators2.csv', parse_dates=['timestamp']).tail(96)
X_test = scaler.transform(test_data.drop(['timestamp', 'close'], axis=1))
y_test = scaler.transform(test_data[['close']]).ravel()

# Create a DataLoader for the test data
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                              torch.tensor(y_test, dtype=torch.float32))
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=96, shuffle=False)

# Put the model in evaluation mode
best_model.eval()

# Generate predicted values for the test data
y_pred = []
with torch.no_grad():
    for x, y in test_data_loader:
        y_pred_batch = best_model(x)
        y_pred_batch = y_pred_batch.cpu().numpy()
        y_pred.extend(y_pred_batch)

# Inverse transform the predicted values using the y_scaler
y_pred = y_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))

# Load the last 96 candles from Binance using ccxt.simulated
exchange = ccxt.simulated()
symbol = 'BTC/USDT'
timeframe = '5m'
candles = exchange.fetch_ohlcv(symbol, timeframe, limit=96)
candles_df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
candles_df['timestamp'] = pd.to_datetime(candles_df['timestamp'], unit='ms')

# Calculate the predicted close using the last 96 candles and the predicted values from the model
last_close = candles_df['close'].iloc[-1]
predicted_close = y_pred[-1][0]

# Print the last close, predicted close, and the difference between the two
print(f"Last close: {last_close:.2f}")
print(f"Predicted close: {predicted_close:.2f}")
print(f"Difference: {predicted_close - last_close:.2f}")
