import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from classdirectory.classfile_test3 import LSTMRegressor
from classdirectory.classfile_test3 import OptunaLSTMRegressor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    df = pd.read_csv("BTC_USDT_1h_with_indicators.csv")
    # Convert the date column to a datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Set the date column as the index
    df.set_index('timestamp', inplace=True)

    features = df.drop(['close'], axis=1).values
    labels = df['close'].values.reshape(-1, 1)

    # scale data
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    labels_scaled = scaler.fit_transform(labels)

    # split dataset into train, val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(features_scaled, labels_scaled, test_size=0.1,
                                                                random_state=42, shuffle=False)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42,
                                                      shuffle=False)

    # convert to tensors
    features_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    labels_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    features_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    labels_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    features_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    labels_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # create datasets
    train_dataset = TensorDataset(features_train_tensor, labels_train_tensor)
    val_dataset = TensorDataset(features_val_tensor, labels_val_tensor)
    test_dataset = TensorDataset(features_test_tensor, labels_test_tensor)

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, drop_last=True)

    # create model
    input_size = features.shape[1]
    hidden_size = 32
    num_layers = 4
    output_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-3
    weight_decay = 1e-4
    dropout = 0.2

    model = LSTMRegressor(input_size, hidden_size, num_layers, output_size,
                          learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout).to(device)
    # get the optimizer and the learning rate scheduler
    optimizer_config = model.configure_optimizers()
    optimizer = optimizer_config['optimizer']
    lr_scheduler = optimizer_config['lr_scheduler']

    # create checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best_model_{epoch}_{val_loss:.4f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=15, mode="min")
    # create hyperparameters dictionary
    # initialize logger
    logger = TensorBoardLogger(save_dir='./lightning_logs', name='bilstm-regressor', default_hp_metric=True)
    # log hyperparameters to TensorBoard
    logger.log_hyperparams({
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "output_size": output_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout
    })
    # train model
    optuna_regressor = OptunaLSTMRegressor(input_size=input_size, hidden_size=64, num_layers=2, output_size=output_size)
    optuna_regressor.train(train_loader, val_loader)
    # load the best model
    optuna_regressor.load_from_checkpoint(optuna_regressor.best_model_path)

    # switch to evaluation mode
    optuna_regressor.eval()
    optuna_regressor.to(device)

