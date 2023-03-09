import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pandas_ta
import ccxt
from sklearn.preprocessing import MinMaxScaler
from classdirectory.classfile_test4 import LSTMRegressor

# create exchange object
exchange = ccxt.binance()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    sliced_rows = 101
    df = df.iloc[sliced_rows:]
    features = df.drop(['close'], axis=1).values
    labels = df['close'].values.reshape(-1, 1)
    # scale data
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    labels_scaled = scaler.fit_transform(labels)

    # set sequence length
    sequence_length = 24

    # create input/output sequences with sliding window method
    inputs = []
    outputs = []
    for i in range(len(features_scaled) - sequence_length):
        inputs.append(features_scaled[i:i + sequence_length])
        outputs.append(labels_scaled[i + sequence_length])

    inputs_array = np.array(inputs, dtype=np.float32)
    outputs_array = np.array(outputs, dtype=np.float32)

    # convert to tensors
    features_test_tensor = torch.tensor(inputs_array, dtype=torch.float32).to(device)
    labels_test_tensor = torch.tensor(outputs_array, dtype=torch.float32).to(device)


    test_dataset = TensorDataset(features_test_tensor, labels_test_tensor)


    test_loader = DataLoader(test_dataset, batch_size=32, drop_last=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create model
    input_size = features.shape[1]
    hidden_size = 64
    num_layers = 2
    output_size = 1
    learning_rate = 0.0001
    weight_decay = 1e-3
    dropout = 0.2
    # load saved model state dictionary
    model_state_dict = torch.load("save/best_model_18_64_2_0.2_20230309-031653-315279.pt", map_location=device)

    # determine the hyperparameters of the saved model by inspecting its state dictionary

    model = LSTMRegressor(input_size, hidden_size, num_layers, output_size, learning_rate, dropout=dropout,
                          weight_decay=weight_decay).to(device)
    model.load_state_dict(model_state_dict)


    # switch to evaluation mode
    model.eval()
    model.to(device)

    # make predictions on test set
    test_pred = []
    test_actual = []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            y_pred = model(x)
            test_pred.append(y_pred.cpu().numpy())
            test_actual.append(y.cpu().numpy())

    # concatenate all batches
    test_pred = np.concatenate(test_pred, axis=0)
    test_actual = np.concatenate(test_actual, axis=0)

    # unscale predictions and actual values
    test_pred_unscaled = scaler.inverse_transform(test_pred.reshape(-1, 1))
    test_actual_unscaled = scaler.inverse_transform(test_actual.reshape(-1, 1))

    # print predicted and actual values

    import matplotlib.pyplot as plt

    # plot predicted and actual values
    test_df = df.tail(len(test_actual_unscaled))

    plt.plot(test_df.index, test_actual_unscaled, label='actual')
    plt.plot(test_df.index, test_pred_unscaled, label='predicted')
    plt.legend()
    plt.show()
    # get the last timestamp, last close value, and last predicted value
    last_timestamp = test_df.index[-1]
    last_close = test_df['close'][-1]
    last_predicted = test_pred_unscaled[-1][0]

    # print the values
    print(f"Last timestamp: {last_timestamp}")
    print(f"Last close value: {last_close}")
    print(f"Last predicted value: {last_predicted}")
