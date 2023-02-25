import pandas as pd
import pandas_ta as ta

# Load data into dataframe
df = pd.read_csv('BTC_USDT_5m_2015-02-01_now_binance.csv', header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Calculate technical indicators and add to dataframe
df.ta.rsi(length=14, append=True)
df.ta.cci(length=20, constant=0.015, append=True)
df.ta.obv(append=True)
df.ta.ppo(append=True)
df.ta.macd(append=True)
df.ta.mom(append=True, length=14)
df.ta.fisher(append=True, length=14)
df.ta.qqe(append=True, length=14)
df.ta.willr(append=True, length=14)
df.ta.ema(append=True, length=14)
df.ta.chop(append=True, length=14)
sliced_rows = 50
df = df.iloc[sliced_rows:]
column_names = df.columns


# Save dataframe to CSV file
df.to_csv('BTC_USDT_5m_with_indicators3.csv', index=False)
