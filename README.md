Neural Network Training in python (unfinished)

Description:

main.py 

This file contains the main script for running the LSTMTrainer and LoadModel classes to train an LSTM model on cryptocurrency price data and make predictions on real data from the Binance exchange. The script loads the data, splits it into training and testing sets, trains the model, saves the trained model, tests the model on the test set, evaluates the model, and makes predictions on real data using the trained model.

models/trainer_model.py

The trainer_model.py script contains a class called LSTMTrainer which is used to train and evaluate a Bidirectional LSTM model for time series prediction.

The load_data() method is used to load data from a pandas DataFrame object. It extracts the target variable (close_pct_change) and the feature variables from the DataFrame, scales them using a MinMaxScaler, and converts them into inputs and outputs for the LSTM model.

The split_data() method is used to split the data into training, validation, and test sets. It also creates PyTorch DataLoader objects for each set.

The configure_model() method is used to create the Bidirectional LSTM model, optimizer, learning rate scheduler, and other PyTorch Lightning callbacks such as checkpointing, early stopping, and model summary. It also sets up a TensorBoard logger for logging model performance.

The train_model() method is used to train the LSTM model using the PyTorch Lightning Trainer object. It takes as input two Boolean variables auto_lr_find and auto_scale_batch_size that control whether to use automatic learning rate finding and batch size scaling respectively.

The save_model() method is used to save the trained model weights to a file.

The test_model() method is used to test the trained model on the test set. It takes as input a checkpoint path indicating which saved model weights to load.

The evaluate_model() method is used to evaluate the trained model on the test set. It makes predictions using the trained model, unscales the predictions and actual values, and plots them against time. It also prints the last timestamp, actual close value, and predicted close value.

models/lstm_model.py

This code defines a PyTorch Lightning module for training a bidirectional LSTM with self-attention on a time series dataset. The LSTM has an input hidden size, number of layers, and output size specified by the user. The module also includes batch normalization, linear layers, layer normalization, and ReLU activation layers. The module uses mean squared error loss with L1 and L2 regularization. The optimizer is Adam with a learning rate and weight decay specified by the user, and the learning rate is reduced on a plateau during training. The module can be trained, validated, and tested using PyTorch Lightning's Trainer class. The code also defines a SelfAttention module that is used in the LSTM module. The data is assumed to be preprocessed and is loaded using PyTorch's DataLoader class.

Additional scripts:

modules/candledownloader.py

CandleDownloader class, is used for downloading candlestick data from binance exchange using the CCXT library. The downloaded data is stored in a CSV file. The script takes several parameters such as the exchange name, trading pair, timeframe, start and end time, batch size, and output file name. If no output file name is provided, the script generates a default name based on the specified parameters.

The download_candles function fetches the candles from the exchange API and writes them to the output file. It uses an in-memory buffer to store the downloaded data and writes it to the file in batches to minimize disk I/O. If the output file already exists, the function resumes the download from the timestamp of the last candle in the file.

The script also includes error handling and request throttling to prevent the script from being blocked by the exchange. Once the download is complete, the script prints the total number of downloaded candles, batches, and the output file location.


modules/indicatorsadder.py

This script loads historical price data for Bitcoin on the Binance exchange, calculates various technical indicators (EMA, RSI, Bollinger Bands, CCI, ADX), and pivot points. It then saves the data, including the calculated indicators and pivots, to a CSV file for further analysis. The script also performs basic error handling to identify any incorrect data in the price data before calculating the indicators.

modules/indicatorschecker.py

This script loads a CSV file containing Bitcoin price data and technical indicators, checks for missing or infinite values, and drops columns with NaN values. The script first prints the column names and then uses NumPy to check for infinite values in the data. It then counts the number of NaN values and prints the rows and columns that contain NaN values. Finally, the script drops columns with NaN values and saves the resulting dataframe to a new CSV file.
