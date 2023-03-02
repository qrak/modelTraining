import ccxt
import torch
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import StandardScaler

from classdirectory.classfile import LSTMNet


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
    df = df.drop(['timestamp', 'close'], axis=1)
    # Print the column names of the df DataFrame
    print(df.columns)

    # Scale the data
    # Load the pre-trained model
    best_model_path = 'save/best_model_29_64_4_0.1.pt'
    best_model = LSTMNet(input_size=29, hidden_size=64, num_layers=4, output_size=1, dropout=0.1)
    best_model.load_state_dict(torch.load(best_model_path))

    scaler = StandardScaler()
    ta_df_scaled = scaler.fit_transform(df)

    # Reshape the data to be compatible with the model
    num_rows, num_cols = ta_df_scaled.shape
    ta_tensor = torch.tensor(ta_df_scaled, dtype=torch.float32).view(1, num_rows, num_cols)

    # Put the model in evaluation mode
    best_model.eval()

    # Generate predicted values for the test data
    with torch.no_grad():
        y_pred = best_model(ta_tensor)

    print("Shape of y_pred:", y_pred.shape)
    # Manually scale the predicted value using the same mean and standard deviation as the scaler
    #y_pred = y_pred.cpu().numpy().squeeze()
    y_pred_np = y_pred.cpu().numpy()
    num_rows = ta_tensor.shape[1] - 20

    y_pred_scaled = (y_pred_np * scaler.scale_[0]) + scaler.mean_[0]
    y_pred_scaled = y_pred_scaled.reshape(-1)

    # Print the shape of y_pred_scaled
    print("Shape of y_pred_scaled:", y_pred_scaled.shape)

    # Print the predictions for the next 20 candles
    last_close = candles_df['close'].iloc[-1]
    print(f"Last close: {last_close:.2f}")
    for i in range(1, len(y_pred_scaled) + 1):
        predicted_close = y_pred_scaled[i - 1]
        print(f"Predicted close for candle {i}: {float(predicted_close):.2f}")
        print(f"Difference for candle {i}: {float(predicted_close - last_close):.2f}")
        last_close = predicted_close