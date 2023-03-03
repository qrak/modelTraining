"""import ccxt
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
    print(test_df)"""

import os
import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import DataLoader, TensorDataset
from classdirectory.classfile_test import LSTMNet
#import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    data = pd.read_csv('BTC_USDT_1h_with_indicators.csv', parse_dates=['timestamp'])
    features = data.drop(['timestamp', 'close'], axis=1)
    target = data['close'].values
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    features_scaled = scaler_features.fit_transform(features)
    target_scaled = scaler_target.fit_transform(target.reshape(-1, 1))

    # Use the last 20% of the preprocessed data as the test set
    test_start_idx = int(len(features) * 0.8)
    X_test = features_scaled[test_start_idx:]
    y_test = target_scaled[test_start_idx:]
    # Convert X_test to a NumPy array
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    scaler_target.fit_transform(target.reshape(-1, 1))
    # Define the hyperparameters to search over
    input_size = 29
    hidden_size = 8
    num_layers = 4
    dropout_size = 0.1


    model = LSTMNet(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                    output_size=1, dropout=dropout_size).to(device)


    best_model_path = f"save/best_model_{input_size}_{hidden_size}_{num_layers}_{dropout_size}.pt"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # Evaluate the best model on the test set
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).to(device),
                                 torch.tensor(y_test, dtype=torch.float32).to(device))

    # Load the best model and its weights
    best_model = LSTMNet(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=1,
                         dropout=dropout_size)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(device)

    # Create a new data loader for the test set
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    # Make predictions on the test set using the best model
    predicted_prices = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = best_model(inputs)
            predicted_prices.extend(outputs.cpu().numpy())

    # Get the actual close prices from the test set
    actual_prices = scaler_target.inverse_transform(y_test).flatten()

    # Sort the predicted and actual prices based on the predicted prices
    sorted_indices = np.argsort(predicted_prices)
    predicted_prices_sorted = np.array(predicted_prices)[sorted_indices]
    actual_prices_sorted = actual_prices[sorted_indices]
    # Sort the predicted and actual prices based on the predicted prices

    # Create the predicted and actual dataframes and print the entire dataframe to the console
    predicted_prices = scaler_target.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()
    predicted_df = pd.DataFrame(predicted_prices, columns=['Predicted Close'])
    actual_df = pd.DataFrame(actual_prices, columns=['Actual Close'])
    results_df = pd.concat([predicted_df, actual_df], axis=1)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(results_df)

