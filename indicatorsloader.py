import pandas as pd
import numpy as np
# Load data into dataframe
df = pd.read_csv('csv/BTC_USDT_15m_2015-01-01_now_binance.csv', header=0)

column_names = df.columns
print(df)
# Print the column names
print(f'Column names: {column_names}')
# Get the values as a numpy array
X = df.iloc[:, 1:].values

# Find the infinite values in X
mask = np.isinf(X)
if mask.any():
    indices = np.where(mask)
    print(f'The following values are infinite:')
    for i, j in zip(*indices):
        print(f'({i}, {j}): {X[i, j]}')
else:
    print(f'There are no infinite values in the data.')

# Check for NaN values and count them
num_nans = df.isna().sum().sum()

# Print the result
print(f'There are {num_nans} NaN values in the DataFrame')

# Check which rows have NaN values
rows_with_nan = df.isna().any(axis=1)
print("Rows with NaN values:\n", df[rows_with_nan])


# Check which columns have NaN values
cols_with_nan = df.columns[df.isna().any()].tolist()
print("Columns with NaN values:", cols_with_nan)

# Drop columns with Nan values
#df = df.dropna(axis=1)

# Save dataframe to CSV file
#df.to_csv('BTC_USDT_5m_with_indicators.csv', index=False)
