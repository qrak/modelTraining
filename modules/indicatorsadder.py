import pandas_ta as ta
import pandas as pd
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


# Load data into dataframe
df = pd.read_csv('../csv_ohlcv/BTC_USDT_1h_2015-01-01_now_binance.csv', header=0, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
close_col = df['close']
for i, val in enumerate(close_col):
    try:
        float_val = float(val)
    except ValueError:
        print(f"Found incorrect value at index {i}: {val}")

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.ta.ema(length=14, append=True)
df.ta.rsi(length=14, append=True)
df.ta.bbands(length=14, append=True)
df.ta.cci(length=14, append=True)
df.ta.adx(length=14, append=True)
df.set_index('timestamp', inplace=True)
df = df.sort_index()
# Add new features to the dataframe
df['day_of_week'] = df.index.dayofweek
df['day_of_month'] = df.index.day
df['day_of_year'] = df.index.dayofyear

# Calculate the percentage change of the 'close' price for each period
df['close_pct_change'] = df['close'].pct_change()

sliced_rows = 50
df = df.iloc[sliced_rows:]
print(df)

#column_names = df.columns

# Save dataframe to CSV file
df.to_csv('../csv_modified/BTC_USDT_1h_indicators.csv', index=True)

