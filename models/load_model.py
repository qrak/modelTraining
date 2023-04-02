import ccxt
import pandas as pd
import pandas_ta as ta
import pytz
from torch.utils.data import DataLoader, TensorDataset
from models.trainer_model import ModelTrainer


class ModelLoader(ModelTrainer):
    def __init__(self, config):
        super().__init__(config)

    def load_and_evaluate(self, file_path, tail_n=200):
        self.load_model(file_path)
        self.test_dataset = TensorDataset(self.inputs_tensor, self.outputs_tensor)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config['batch_size'], shuffle=False, drop_last=True)
        self.evaluate_model(tail_n)

    def fetch_live_data(self):
        exchange = ccxt.binance()
        symbol = 'BTC/USDT'
        timeframe = '5m'
        limit = 1000
        self.logger.info(
            f"Fetching live data from exchange for symbol: {symbol}, timeframe: {timeframe}, candles: {limit}.")
        ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        self.df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        # Convert timestamp timezone to +2 hours
        self.df['timestamp'] = self.df['timestamp'].apply(lambda x: ModelLoader.convert_timezone(x, 'UTC', 'Etc/GMT-2'))
        self.df.set_index('timestamp', inplace=True)
        self.df.ta.bop(append=True, length=24)
        self.df.ta.cfo(append=True, length=24)
        self.df.ta.psar(append=True, length=24)
        self.df.ta.natr(append=True, length=24)
        self.df.ta.eri(append=True, length=24)
        self.df.ta.fisher(append=True, length=24)
        self.df.ta.dm(append=True, length=24)
        self.df.ta.kdj(append=True, length=24)
        self.df.ta.pgo(append=True, length=24)
        self.df.ta.willr(append=True, length=24)
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['day_of_month'] = self.df.index.day
        self.df['day_of_year'] = self.df.index.dayofyear
        sliced_rows = 40
        self.df = self.df.iloc[sliced_rows:]
        self.df = self.df.dropna(axis=1)
        input_size = self.df.shape[1]
        return self.df, input_size

    @staticmethod
    def convert_timezone(timestamp, from_tz, to_tz):
        from_tz = pytz.timezone(from_tz)
        to_tz = pytz.timezone(to_tz)
        timestamp = from_tz.localize(timestamp)
        return timestamp.astimezone(to_tz)

