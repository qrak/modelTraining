import pandas as pd
from models.trainer_model import LSTMTrainer
from models.load_model import LoadModel

if __name__ == '__main__':
    # input_size is the number of features you have in your csv_ohlcv file #input_size variable is taken from csv_ohlcv columns
    lstm_trainer = LSTMTrainer(hidden_size=64, num_layers=2, output_size=1, learning_rate=0.0001,
    dropout=0.2, weight_decay=1e-3, num_epochs=200)
    try:
        lstm_trainer.load_data(pd.read_csv("csv_modified/BTC_USDT_1h_indicators.csv"))
    except FileNotFoundError:
        print('Wrong csv_ohlcv filename, load your csv_ohlcv file first!')
        exit()
    lstm_trainer.split_data()
    lstm_trainer.configure_model()
    lstm_trainer.train_model()
    lstm_trainer.save_model()
    lstm_trainer.test_model()
    lstm_trainer.evaluate_model()
    lstm_model_loader = LoadModel()
    # real data fetch not finished here, for real data check main_load.py
    lstm_model_loader.predict()