import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pandas_ta
import ccxt
from sklearn.preprocessing import MinMaxScaler
from classdirectory.classfile import LSTMRegressor

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
    df.set_index('timestamp', inplace=True)
    # Calculate technical indicators and add to dataframe
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=9, append=True)
    df.ta.rsi(length=21, append=True)
    df.ta.rsi(length=30, append=True)
    df.ta.bbands(length=9, append=True)
    df.ta.bbands(length=21, append=True)
    df.ta.bbands(length=30, append=True)
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


    # convert to tensors
    features_test_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    labels_test_tensor = torch.tensor(labels_scaled, dtype=torch.float32).to(device)


    test_dataset = TensorDataset(features_test_tensor, labels_test_tensor)


    test_loader = DataLoader(test_dataset, batch_size=16, drop_last=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create model
    input_size = features.shape[1]
    hidden_size = 32
    num_layers = 4
    output_size = 1
    
    learning_rate = 1e-3
    weight_decay = 1e-4
    dropout = 0.2
    # load saved model state dictionary
    model_state_dict = torch.load("save/best_model_32_64_3_0.2.pt", map_location=device)

    # determine the hyperparameters of the saved model by inspecting its state dictionary

    model = LSTMRegressor(input_size, hidden_size, num_layers, output_size,
                          learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout).to(device)
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
