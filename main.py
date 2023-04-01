from models.load_model import LoadModel
from models.trainer_model import LSTMTrainer


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

    trainer = LSTMTrainer(config)
    #trainer.preprocess_data("csv_ohlcv/BTC_USDT_1m_2015-01-01_now_binance.csv", chunksize=10000)
    trainer.preprocess_data("csv_modified/BTC_USDT_5m_indicators.csv", chunksize=10000, input_type='file')
    trainer.split_data(test_size=0.1, random_state=42)
    trainer.configure_model()
    trainer.train_model()
    trainer.save_model()
    trainer.test_model()
    trainer.evaluate_model(tail_n=200)
