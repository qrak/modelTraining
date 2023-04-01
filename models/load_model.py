import pandas as pd
import pandas_ta as ta
import ccxt
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.trainer_model import LSTMTrainer

class LoadModel(LSTMTrainer):
    def __init__(self, file_path: str, input_size: int, hidden_size: int, num_layers: int, dropout: float, batch_size: int, sequence_length: int, config: dict):
        super().__init__(config)
        self.file_path = file_path
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.config['sequence_length'] = sequence_length
        self.model = BitcoinPredictor(input_size=self.input_size, hidden_size=self.hidden_size, output_size=1, num_layers=self.num_layers, num_heads=1, sequence_length=self.sequence_length, batch_size=self.batch_size, num_epochs=self.config['num_epochs'], learning_rate=self.config['learning_rate'], weight_decay=self.config['weight_decay'], dropout=self.dropout)
        self.load_model()
    def predict(self):
        exchange = ccxt.binance()
        symbol = 'BTC/USDT'
        timeframe = '1h'
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

        sliced_rows = 50
        self.df = self.df.iloc[sliced_rows:]
        self.preprocess_data(self.df, chunksize=10000)

        # Create tensors for real-time data
        self.preprocess_data(self.df, chunksize=None)
        self.test_dataset = TensorDataset(self.inputs_tensor, self.outputs_tensor)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config['batch_size'], shuffle=False,
                                      drop_last=True, num_workers=4)
        self.evaluate_model()
