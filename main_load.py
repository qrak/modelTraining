import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pandas_ta
import ccxt
from sklearn.preprocessing import MinMaxScaler
from classdirectory.classfile_test3 import LSTMRegressor

# create exchange object
exchange = ccxt.binance()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
def calculate_pivots(df):

    window = 24
    # Calculate pivot points for each 24-hour period
    df['pivot'] = df['high'].rolling(window=window).sum() + df['low'].rolling(window=window).sum() + df[
        'close'].rolling(window=window).sum()
    df['pivot'] = df['pivot'] / (3 * window)

    # Calculate support and resistance levels for each 24-hour period
    df['s1'] = (2 * df['pivot'].rolling(window=window).sum()) - df['high'].rolling(window=window).max()
    df['s2'] = df['pivot'].rolling(window=window).sum() - (
                df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min())
    df['s3'] = df['low'].rolling(window=window).min() - 2 * (
                df['high'].rolling(window=window).max() - df['pivot'].rolling(window=window).sum())
    df['s4'] = df['s3'] - (df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min())
    df['r1'] = (2 * df['pivot'].rolling(window=window).sum()) - df['low'].rolling(window=window).min()
    df['r2'] = df['pivot'].rolling(window=window).sum() + (
                df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min())
    df['r3'] = df['high'].rolling(window=window).max() + 2 * (
                df['pivot'].rolling(window=window).sum() - df['low'].rolling(window=window).min())
    df['r4'] = df['r3'] + (df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min())


if __name__ == '__main__':

    # load data from exchange API
    symbol = 'BTC/USDT'
    timeframe = '1h'
    limit = 1000  # number of candles to retrieve
    ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
    # convert data to dataframe
    df = pd.DataFrame(ohlcv[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp', ascending=True)
    df.set_index('timestamp', inplace=True)
    # Create new columns for day of week, day of month, and day of year
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['day_of_year'] = df.index.dayofyear

    # Calculate technical indicators and add to dataframe
    df.ta.ema(length=14, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.bbands(length=14, append=True)
    df.ta.cci(length=14, append=True)
    df.ta.adx(length=14, append=True)
    calculate_pivots(df)
    sliced_rows = 101
    df = df.iloc[sliced_rows:]

    # Print the number of rows in the dataframe
    print(f'Number of rows: {df.shape[0]}')
    model_state_dict = torch.load("save/best_model_27_96_2_0.2_20230309-194906-808303.pt", map_location=device)

    # Create a new instance of the LSTM model with the same hyperparameters as the pre-trained model
    input_size = df.shape[1] - 1  # exclude the 'close' column
    hidden_size = 96
    num_layers = 2
    output_size = 1
    learning_rate = 0.0001
    weight_decay = 1e-3
    dropout = 0.2
    sequence_length = 24
    model = LSTMRegressor(input_size, hidden_size, num_layers, output_size, learning_rate, dropout=dropout,
                          weight_decay=weight_decay).to(device)

    # Load the state dictionary into the new model
    model.load_state_dict(model_state_dict)

    # Convert the new data into the same format as the training data
    features_new = df.drop(['close'], axis=1).values
    labels_new = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    features_scaled_new = scaler.fit_transform(features_new)
    labels_scaled_new = scaler.fit_transform(labels_new)

    # Create input/output sequences with sliding window method
    inputs_new = []
    outputs_new = []
    for i in range(len(features_scaled_new) - sequence_length):
        inputs_new.append(features_scaled_new[i:i + sequence_length])
        outputs_new.append(labels_scaled_new[i + sequence_length])
    inputs_array_new = np.array(inputs_new, dtype=np.float32)
    outputs_array_new = np.array(outputs_new, dtype=np.float32)

    # Convert to tensors
    features_new_tensor = torch.tensor(inputs_array_new, dtype=torch.float32).to(device)
    labels_new_tensor = torch.tensor(outputs_array_new, dtype=torch.float32).to(device)

    # Pass the new data through the model to generate predictions
    model.eval()
    model.to(device)
    with torch.no_grad():
        predicted_labels_new_tensor = model(features_new_tensor)

    # Convert the predicted labels back to their original scale
    predicted_labels_new = scaler.inverse_transform(predicted_labels_new_tensor.cpu().numpy())

    # Print the actual values from live data and the predicted values
    actual_labels_new = labels_new[sequence_length:]
    print("Actual close prices:", actual_labels_new)
    print("Predicted close prices:", predicted_labels_new.flatten())
    import matplotlib.pyplot as plt

    # Plot the actual and predicted close prices
    plt.figure(figsize=(12, 8))
    plt.plot(actual_labels_new, label='Actual Close Prices')
    plt.plot(predicted_labels_new.flatten(), label='Predicted Close Prices')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.title('Actual vs. Predicted Close Prices')
    plt.legend()
    plt.show()
