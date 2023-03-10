import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pandas_ta
import ccxt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from classdirectory.lstm_model import LSTMRegressor
import matplotlib.pyplot as plt
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
    limit = 1000 # number of candles to retrieve
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
    df_copy = df.copy()
    sliced_rows = 14
    df = df.iloc[sliced_rows:]
    # Print the number of rows in the dataframe
    print(f'Number of rows: {df.shape[0]}')
    model_state_dict = torch.load("save/best_model_10_8_2_0.2_20230310-045003-736073.pt", map_location=device)
    # Convert the new data into the same format as the training data
    features_new = df.drop(['close'], axis=1).values
    labels_new = df['close'].values.reshape(-1, 1)
    scaler = StandardScaler()
    features_scaled_new = scaler.fit_transform(features_new)
    labels_scaled_new = scaler.fit_transform(labels_new)
    sequence_length = 24
    # Create input/output sequences with sliding window method
    inputs_new = []
    outputs_new = []
    for i in range(len(features_scaled_new) - sequence_length):
        inputs_new.append(features_scaled_new[i:i + sequence_length])
        outputs_new.append(labels_scaled_new[i + sequence_length])
    inputs_array_new = np.array(inputs_new, dtype=np.float32)
    outputs_array_new = np.array(outputs_new, dtype=np.float32)
    print(inputs_array_new.shape)
    print(outputs_array_new.shape)
    # Create a TensorDataset from inputs and outputs
    new_dataset = TensorDataset(torch.tensor(inputs_array_new), torch.tensor(outputs_array_new))

    # Create a DataLoader to iterate over the data
    batch_size = 32
    new_dataloader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

    # Create a new instance of the LSTM model with the same hyperparameters as the pre-trained model
    input_size = features_new.shape[1]  # exclude the 'close' column
    hidden_size = 8
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
    # Pass the new data through the model to generate predictions

    model.to(device)
    model.eval()
    # Get the predictions for each batch in the new dataloader
    predictions = []
    for i in range(sequence_length, len(df)):
        # Get the input sequence for the current prediction
        inputs = features_scaled_new[i - sequence_length:i].reshape(1, sequence_length, -1)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)

        # Make a prediction for the current `close` value
        with torch.no_grad():
            prediction = model(inputs_tensor).cpu().numpy()

        # Invert the scaling to get the actual predicted value
        prediction = scaler.inverse_transform(prediction)[0][0]

        predictions.append(prediction)

    # Plot the actual close and predicted close
    plt.figure(figsize=(10, 6))
    plt.plot(df_copy.index[sliced_rows + sequence_length:len(predictions) + sliced_rows + sequence_length],
             df_copy['close'][sliced_rows + sequence_length:], label='Actual Close')
    plt.plot(df_copy.index[sliced_rows + sequence_length:len(predictions) + sliced_rows + sequence_length], predictions,
             label='Predicted Close')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Print the last actual value and the predicted values
    last_row = df.iloc[-1]
    print(f"Last actual close price: {last_row['close']}")
    print(f"Last predicted close price: {predictions[-1]}")