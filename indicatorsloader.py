import pandas as pd

# Load data into dataframe
df = pd.read_csv('BTC_USDT_5m_with_indicators3.csv', header=0)

column_names = df.columns

# Print the column names
print(f'Column names: {column_names}')

# Check for NaN values and count them
num_nans = df.isna().sum().sum()

# Print the result
print(f'There are {num_nans} NaN values in the DataFrame')

# Check which rows have NaN values
rows_with_nan = df.isna().any(axis=1)
print("Rows with NaN values:\n", df[rows_with_nan])

# Drop columns with Nan values
df = df.dropna(axis=1)

# Check which columns have NaN values
cols_with_nan = df.columns[df.isna().any()].tolist()
print("Columns with NaN values:", cols_with_nan)


# Save dataframe to CSV file
#df.to_csv('BTC_USDT_5m_with_indicators2.csv', index=False)
