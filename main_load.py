import ccxt
import torch
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from classdirectory.classfile_test import LSTMNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':


    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1h'
    limit = 1000  # Maximum value allowed by Binance
    candles = []

    while True:
        # Fetch the candles
        new_candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=None)
        # Stop fetching if there are no new candles
        if len(new_candles) == 0:
            break
        # Append the new candles to the existing list of candles
        candles += new_candles
        # Set the `since` parameter for the next fetch to the timestamp of the last candle fetched
        last_timestamp = new_candles[-1][0]
        limit -= len(new_candles)
        if limit <= 0:
            break

    candles_df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    candles_df['timestamp'] = pd.to_datetime(candles_df['timestamp'], unit='ms')

    # Calculate technical indicators and add to dataframe
    candles_df.ta.ema(length=20, append=True)
    candles_df.ta.ema(length=50, append=True)
    candles_df.ta.ema(length=100, append=True)
    candles_df.ta.rsi(length=9, append=True)
    candles_df.ta.rsi(length=21, append=True)
    candles_df.ta.rsi(length=30, append=True)
    candles_df.ta.bbands(length=9, append=True)
    candles_df.ta.bbands(length=21, append=True)
    candles_df.ta.bbands(length=30, append=True)
    candles_df.ta.cci(length=14, append=True)
    candles_df.ta.adx(length=14, append=True)
    sliced_rows = 150
    df = candles_df.iloc[sliced_rows:]
    df = df.drop('timestamp', axis=1)

    # Print the column names of the df DataFrame

    # Convert the DataFrame to a PyTorch DataLoader
    batch_size = 64

    # Pass only the technical indicators and other features to the first tensor, and not the target variable ('close')
    inputs = torch.tensor(df.drop('close', axis=1).values, dtype=torch.float32)
    labels = torch.tensor(df['close'].values, dtype=torch.float32).to(
        device)  # Move the target tensor to the same device as the input tensor
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load the pre-trained model
    best_model_path = 'save/best_model_29_8_4_0.1.pt'
    best_model = LSTMNet(input_size=29, hidden_size=8, num_layers=4, output_size=1, dropout=0.1)
    # Move the model to the same device as the input tensor
    best_model.to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    target = candles_df['close'].values.reshape(-1, 1)


    # Make predictions on the test set using the best model
    # Make predictions on the test set using the best model
    with torch.no_grad():
        best_model.eval()
        # Pass the input tensor `inputs` to the LSTM model instead of `candles_df`
        y_pred = best_model(inputs.to(device)).cpu().numpy()

    # Inverse transform the predicted and actual close values
    scaler = MinMaxScaler()
    # Fit the scaler on the training data
    scaler.fit(inputs_train.numpy())
    # Scale the inputs and labels for both training and testing
    inputs_train = scaler.transform(inputs_train.numpy())
    inputs_test = scaler.transform(inputs_test.numpy())
    labels_train = scaler.transform(labels_train.numpy())
    labels_test = scaler.transform(labels_test.numpy())
    y_pred = scaler.inverse_transform(y_pred)

    # Create a DataFrame with the actual and predicted close values for the test set
    test_df = pd.DataFrame({'Actual Close': df['close'], 'Predicted Close': y_pred.flatten()})
    print(test_df)

