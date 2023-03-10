Neural Network Training in python (unfinished)

Description:

main.py and classdirectory/lstm_model.py

These two files contain code for training and evaluating a neural network for time series regression using PyTorch and PyTorch Lightning. The main.py file defines a dataset class that loads time series data from a CSV file and preprocesses it for use in the neural network. The LSTMRegressor class in the lstm_model.py file defines the neural network architecture, which consists of an LSTM layer, followed by batch normalization, self-attention, layer normalization, a fully connected layer with ReLU activation, and a final linear layer. The model also incorporates L1 and L2 regularization, as well as dropout for regularization during training. The LSTMRegressor class inherits from PyTorch Lightning's LightningModule class, which provides additional functionality for training and evaluating the model, such as automatic logging of metrics and early stopping.

The main.py file contains the main training script, which uses PyTorch Lightning's Trainer class to manage the training process. The script loads the data from a CSV file, splits it into training, validation, and test sets, and trains the LSTMRegressor model on the training set. During training, the script logs the training loss, validation loss, and learning rate, and uses a learning rate scheduler to adjust the learning rate based on the validation loss. After training, the script evaluates the model on the test set and logs the test loss.

Together, these files provide a complete example of how to build and train a neural network for time series regression using PyTorch and PyTorch Lightning.

main_load.py

This script fetches OHLCV data from the Binance exchange, calculates technical indicators and pivot points for the cryptocurrency BTC/USDT, and uses a pre-trained LSTM model to generate predictions for future closing prices. The data is first pre-processed by standardizing the features and target variable. The script then uses a sliding window method to create input/output sequences for the LSTM model. Finally, the model is used to generate predictions for the new data, which are plotted against the actual closing prices. The last actual close price and last predicted close price are printed for reference.

Additional scripts:

candledownloader.py

CandleDownloader class, is used for downloading candlestick data from binance exchange using the CCXT library. The downloaded data is stored in a CSV file. The script takes several parameters such as the exchange name, trading pair, timeframe, start and end time, batch size, and output file name. If no output file name is provided, the script generates a default name based on the specified parameters.

The download_candles function fetches the candles from the exchange API and writes them to the output file. It uses an in-memory buffer to store the downloaded data and writes it to the file in batches to minimize disk I/O. If the output file already exists, the function resumes the download from the timestamp of the last candle in the file.

The script also includes error handling and request throttling to prevent the script from being blocked by the exchange. Once the download is complete, the script prints the total number of downloaded candles, batches, and the output file location.


indicatorsadder.py

This script loads historical price data for Bitcoin on the Binance exchange, calculates various technical indicators (EMA, RSI, Bollinger Bands, CCI, ADX), and pivot points. It then saves the data, including the calculated indicators and pivots, to a CSV file for further analysis. The script also performs basic error handling to identify any incorrect data in the price data before calculating the indicators.

indicatorschecker.py

This script loads a CSV file containing Bitcoin price data and technical indicators, checks for missing or infinite values, and drops columns with NaN values. The script first prints the column names and then uses NumPy to check for infinite values in the data. It then counts the number of NaN values and prints the rows and columns that contain NaN values. Finally, the script drops columns with NaN values and saves the resulting dataframe to a new CSV file.
