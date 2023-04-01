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
df = pd.read_csv('../csv_ohlcv/BTC_USDT_5m_2015-01-01_now_binance.csv', header=0, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
close_col = df['close']
for i, val in enumerate(close_col):
    try:
        float_val = float(val)
    except ValueError:
        print(f"Found incorrect value at index {i}: {val}")

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.ta.bop(append=True, length=24) #BOP: 0.4038390815258026
df.ta.cfo(append=True, length=24) #CFO_9: 0.4027763307094574
df.ta.psar(append=True, length=24) #PSARr_0.02_0.2: 0.23676089942455292
df.ta.natr(append=True, length=24) #NATR_14: 0.1687399297952652
df.ta.eri(append=True, length=24) #eri:0.1
df.ta.fisher(append=True, length=24)
df.ta.dm(append=True, length=24)
df.ta.kdj(append=True, length=24)
df.ta.pgo(append=True, length=24)
df.ta.willr(append=True, length=24)
df.set_index('timestamp', inplace=True)
df = df.sort_index()
# Add new features to the dataframe
df['day_of_week'] = df.index.dayofweek
df['day_of_month'] = df.index.day
df['day_of_year'] = df.index.dayofyear

# Calculate the percentage change of the 'close' price for each period
#df['close_pct_change'] = df['close'].pct_change()

sliced_rows = 40
df = df.iloc[sliced_rows:]
print(df)

#column_names = df.columns

# Save dataframe to CSV file
df.to_csv('../csv_modified/BTC_USDT_5m_indicators.csv', index=True)

