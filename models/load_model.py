import pandas as pd
import pandas_ta as ta
import ccxt
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.trainer_model import LSTMTrainer
class LoadModel(LSTMTrainer):

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def dir_path(self):
        dir_path = self.dir_path
        return dir_path
    def predict(self):
        exchange = ccxt.binance()
        symbol = 'BTC/USDT'
        timeframe = '1h'
        limit = 1000  # number of candles to retrieve
        ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        self.df = pd.DataFrame(ohlcv[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        self.df.ta.ema(length=14, append=True)
        self.df.ta.rsi(length=14, append=True)
        self.df.ta.bbands(length=14, append=True)
        self.df.ta.cci(length=14, append=True)
        self.df.ta.adx(length=14, append=True)
        self.df.set_index('timestamp', inplace=True)
        self.df = self.df.sort_index()
        # Add new features to the dataframe
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['day_of_month'] = self.df.index.day
        self.df['day_of_year'] = self.df.index.dayofyear
        # Calculate the percentage change of the 'close' price for each period
        self.df['close_pct_change'] = self.df['close'].pct_change()
        model_state_dict = torch.load(self.file_path, map_location=self.device)
        sliced_rows = 50

        self.df = self.df.iloc[sliced_rows:]
        self.preprocess_data(self.df)
        self.configure_model()
        self.model.load_state_dict(model_state_dict)
        self.test_dataset = TensorDataset(self.inputs_tensor, self.outputs_tensor)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.evaluate_model()
