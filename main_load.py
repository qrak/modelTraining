from models.load_model import LoadModel

if __name__ == '__main__':
    # real data test on binance exchange
    lstm_model_loader = LoadModel(file_path='save/best_model_18_64_2_0.2_20230311-213020-349177.pt')
    lstm_model_loader.predict()