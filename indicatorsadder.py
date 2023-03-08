import pandas as pd
import pandas_ta as ta

# Load data into dataframe
df = pd.read_csv('csv/BTC_USDT_15m_2015-01-01_now_binance.csv', header=0, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
close_col = df['close']
for i, val in enumerate(close_col):
    try:
        float_val = float(val)
    except ValueError:
        print(f"Found incorrect value at index {i}: {val}")
# Calculate technical indicators and add to dataframe
df.ta.ema(length=14, append=True)
df.ta.rsi(length=14, append=True)
df.ta.bbands(length=14, append=True)
df.ta.cci(length=14, append=True)
df.ta.adx(length=14, append=True)
sliced_rows = 101
df = df.iloc[sliced_rows:]

#column_names = df.columns

# Save dataframe to CSV file
df.to_csv('BTC_USDT_15m_with_indicators.csv', index=False)
