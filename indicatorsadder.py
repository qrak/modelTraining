import pandas as pd
import pandas_ta as ta

# Load data into dataframe
df = pd.read_csv('BTC_USDT_5m_2015-02-01_now_binance.csv', header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Calculate technical indicators and add to dataframe
df.ta.adx(length=14, append=True)
df.ta.rsi(length=14, append=True)
df.ta.cci(length=20, constant=0.015, append=True)
# Compute On-Balance Volume
df.ta.obv(append=True)

# Compute Percentage Price Oscillator
df.ta.ppo(append=True)
df = df.iloc[30:]

# Save dataframe to CSV file
df.to_csv('BTC_USDT_5m_with_indicators2.csv', index=False)
