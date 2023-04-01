from models.trainer_model import LSTMTrainer
import ccxt
import pandas as pd
import pandas_ta as ta
class ModelLoader(LSTMTrainer):
    def __init__(self, config):
        super().__init__(config)

    def load_and_evaluate(self, file_path, tail_n=200):
        self.load_model(file_path)
        self.evaluate_model(tail_n)

    def fetch_live_data(self):
        exchange = ccxt.binance()
        symbol = 'BTC/USDT'
        timeframe = '5m'
        limit = 1000  # number of candles to retrieve
        ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        self.df = pd.DataFrame(ohlcv[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
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


if __name__ == "__main__":
    config = {
        "hidden_size": 128,
        "num_layers": 2,
        "num_heads": 2,
        "output_size": 1,
        "learning_rate": 0.0001,
        "weight_decay": 1e-3,
        "dropout": 0.2,
        "sequence_length": 24,
        "batch_size": 128,
        "num_epochs": 5
    }

    model_loader = ModelLoader(config)
    data, input_size = model_loader.fetch_live_data()
    config['input_size'] = input_size
    model_loader.preprocess_data(data, chunksize=1000, input_type='dataframe')
    model_loader.configure_model()

    model_loader.load_and_evaluate('save/best_model_128_2_0.2_20230402-000701-296479.pt', tail_n=200)
