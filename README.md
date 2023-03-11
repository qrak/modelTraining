# Neural Network Training in Python

This repository contains code for training neural networks on various datasets. 

## Usage

### LSTM Trainer

The `main.py` script can be used to train a Bidirectional LSTM model on cryptocurrency price data and make predictions on real data from the Binance exchange. The script loads the data, splits it into training and testing sets, trains the model, saves the trained model, tests the model on the test set, evaluates the model, and makes predictions on real data using the trained model.

### LSTM Model

The `lstm_model.py` file defines a PyTorch Lightning module for training a Bidirectional LSTM with self-attention on a time series dataset. The LSTM has an input hidden size, number of layers, and output size specified by the user. The module also includes batch normalization, linear layers, layer normalization, and ReLU activation layers. The module uses mean squared error loss with L1 and L2 regularization. The optimizer is Adam with a learning rate and weight decay specified by the user, and the learning rate is reduced on a plateau during training. The module can be trained, validated, and tested using PyTorch Lightning's Trainer class.

### Candle Downloader

The `candledownloader.py` module can be used to download candlestick data from the Binance exchange using the CCXT library. The downloaded data is stored in a CSV file. The script takes several parameters such as the exchange name, trading pair, timeframe, start and end time, batch size, and output file name. If no output file name is provided, the script generates a default name based on the specified parameters.

### Indicators Adder

The `indicatorsadder.py` module loads historical price data for Bitcoin on the Binance exchange, calculates various technical indicators (EMA, RSI, Bollinger Bands, CCI, ADX), and pivot points. It then saves the data, including the calculated indicators and pivots, to a CSV file for further analysis.

### Indicators Checker

The `indicatorschecker.py` module loads a CSV file containing Bitcoin price data and technical indicators, checks for missing or infinite values, and drops columns with NaN values. The script first prints the column names and then uses NumPy to check for infinite values in the data. It then counts the number of NaN values and prints the rows and columns that contain NaN values. Finally, the script drops columns with NaN values and saves the resulting dataframe to a new CSV file.

## Installation

Clone the repository:

```bash
git clone https://github.com/qrak/neural-network-training.git
cd neural-network-training

```
Create a virtual environment and install the dependencies:

```bash
bash
Copy code
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```