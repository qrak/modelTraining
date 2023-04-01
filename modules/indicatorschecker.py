import pandas as pd
import numpy as np
# Load data into dataframe
df = pd.read_csv('../csv_modified/BTC_USDT_5m_indicators.csv', header=0)
df = df.dropna(axis=1)
column_names = df.columns

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

# Print the number of rows in the dataframe
print(f'Number of rows: {df.shape[0]}')

# Check which columns have NaN values
cols_with_nan = df.columns[df.isna().any()].tolist()
print("Columns with NaN values:", cols_with_nan)



# Save dataframe to CSV file
df.to_csv('../csv_modified/BTC_USDT_5m_indicators.csv', index=False)
