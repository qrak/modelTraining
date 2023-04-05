## Neural Network Training in Python

This repository contains code for training neural networks on various datasets. 

![Neural Networks](image001.png)

## Table of Contents

- [Usage](#usage)
  - [LSTM GUI](#lstm-gui)
  - [LSTM Model](#lstm-model)
  - [Candle Downloader](#candle-downloader)
  - [Indicators Adder](#indicators-adder)
  - [Indicators Checker](#indicators-checker)
- [Installation](#installation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Usage

### LSTM GUI

`main.py` creates a simple GUI application with tkinter that allows users to train and evaluate a crypto/stock price predictor model. The script uses two main classes, `ModelTrainer` and `ModelLoader`, for training and evaluation.

#### Train Model

1. Click the "Train Model" button.
2. Select a CSV file containing the training data.
3. The data is preprocessed, split, and used to train the model.
4. The model is saved to a file and evaluated.

#### Load Model

1. Click the "Load Model" button.
2. Select a previously saved model file.
3. The model is loaded and evaluated on live data.

#### Model Architecture

1. Bidirectional LSTM layer
2. Multi-head self-attention layer
3. Dropout layer
4. Linear output layer

### Candle Downloader

`candledownloader.py` downloads candlestick data from Binance using the CCXT library and saves it to a CSV file. Parameters include exchange name, trading pair, timeframe, start and end time, batch size, and output file name.

### Indicators Adder

`indicatorsadder.py` loads historical price data, calculates technical indicators (EMA, RSI, Bollinger Bands, CCI, ADX), and pivot points, and saves the data to a CSV file.

### Indicators Checker

`indicatorschecker.py` loads a CSV file containing price data and technical indicators, checks for missing or infinite values, drops columns with NaN values, and saves the cleaned data to a new CSV file.

## Installation

Clone the repository:

```bash
git clone https://github.com/qrak/modelTraining.git
cd modelTraining


```
Create a virtual environment and install the dependencies:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

This project is licensed under the GNU General Public License v3.0. You can find a copy of the license in the LICENSE file.

This means that you are free to use, modify, and distribute this software, as long as any modifications you make are also released under the GPL. If you distribute a modified version of this software, you must make the source code available under the same terms as this license.

For more information about the GPL, please see https://www.gnu.org/licenses/gpl-3.0.en.html.

This project was made with the help of ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture. ChatGPT was used to generate text and assist in various aspects of the project's development.
