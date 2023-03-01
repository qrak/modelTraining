import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np

import torch
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from classdirectory.classfile import LSTMNet

# Connect to the Binance exchange API
exchange = ccxt.binance()

# Define the number of historical candlesticks to retrieve
num_candles = 1000

# Retrieve the historical candlestick data
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '5m', limit=num_candles)

# Convert the data to a pandas dataframe
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'], index=None)
# Calculate technical indicators and add to dataframe
df.ta.rsi(length=14, append=True)
df.ta.cci(length=14, constant=0.015, append=True)
df.ta.obv(append=True)
df.ta.ppo(append=True)
df.ta.macd(append=True)
df.ta.mom(length=14, append=True)
df.ta.fisher(length=14, append=True)
df = df.replace([np.inf, -np.inf], np.nan).dropna(how='any')

scaler = StandardScaler()
# Extract the target variable (close price)
X = df.drop(['close', 'timestamp'], axis=1).values
y = df['close'].values
X = scaler.fit_transform(X)
# Perform feature selection
k = 15 # Number of features to select
selector = SelectKBest(f_regression, k=k)
X = selector.fit_transform(X, y)
# Get the indices of the selected features
selected_indices = selector.get_support(indices=True)
# Get the names of the selected features
feature_names = df.drop(['close', 'timestamp'], axis=1).columns
selected_feature_names = feature_names[selected_indices]
# Print the names of the selected features
print(f"The top {k} selected features are:")
for feature in selected_feature_names:
    print(feature)

# Load the best model
saved_model_path = 'best_model.pt'
best_model = LSTMNet(input_size=15, hidden_size=32, num_layers=2, output_size=1, dropout=0.1)
best_model.to(device)

best_model.load_state_dict(torch.load(saved_model_path))
best_model.eval()

# Pass the preprocessed data through the model to get the predicted price
with torch.no_grad():
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_pred = best_model(X_tensor)
    y_pred = y_pred.cpu().numpy()

# Print the predicted price
print('Predicted price:', y_pred)
