import torch
import numpy as np
import pandas as pd
import os
import datetime
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from classdirectory.classfile_test3 import LSTMRegressor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data from CSV file
    df = pd.read_csv("BTC_USDT_15m_with_indicators.csv")

    # Convert the date column to a datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp', ascending=True)
    df.set_index('timestamp', inplace=True)

    # Create new columns for day of week, day of month, and day of year
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['day_of_year'] = df.index.dayofyear

    # separate features and labels
    features = df.drop(columns=['close']).values
    labels = df['close'].values.reshape(-1, 1)

    # scale data
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    labels_scaled = scaler.fit_transform(labels)

    # set sequence length
    sequence_length = 8

    # create input/output sequences with sliding window method
    inputs = []
    outputs = []
    for i in range(len(features_scaled) - sequence_length):
        inputs.append(features_scaled[i:i + sequence_length])
        outputs.append(labels_scaled[i + sequence_length])

    inputs_array = np.array(inputs, dtype=np.float32)
    outputs_array = np.array(outputs, dtype=np.float32)

    # convert input/output sequences to PyTorch tensors
    inputs_tensor = torch.tensor(inputs_array, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs_array, dtype=torch.float32)

    # split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(inputs_tensor, outputs_tensor, test_size=0.1,
                                                                random_state=42, shuffle=False)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42,
                                                      shuffle=False)

    # create TensorDataset objects for train, validation, and test sets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # create DataLoader objects for train, validation, and test sets
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    input_size = features.shape[1]
    hidden_size = 64
    num_layers = 2
    output_size = 1

    learning_rate = 0.0001
    weight_decay = 1e-3
    dropout = 0.2

    model = LSTMRegressor(input_size, hidden_size, num_layers, output_size,
                          learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout).to(device)



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
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=30, mode="min")
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
    trainer = pl.Trainer(max_epochs=200, accelerator="gpu" if torch.cuda.is_available() else 0,
                         logger=logger, log_every_n_steps=1,
                         callbacks=[checkpoint_callback, early_stopping_callback], auto_lr_find=True, auto_scale_batch_size=True)
    model.to(device)
    trainer.fit(model, train_loader, val_loader)

    # Generate file name with current time
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    file_name = f"best_model_{input_size}_{hidden_size}_{num_layers}_{dropout}_{time_stamp}.pt"
    file_path = os.path.join("save", file_name)

    # Save model to file
    torch.save(model.state_dict(), file_path)
    # switch to evaluation mode
    model.eval()
    model.to(device)
    # make predictions on test set
    test_pred = []
    test_actual = []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            y_pred = model(x)
            test_pred.append(y_pred.cpu().numpy())
            test_actual.append(y.cpu().numpy())

    # concatenate all batches
    test_pred = np.concatenate(test_pred, axis=0)
    test_actual = np.concatenate(test_actual, axis=0)

    # unscale predictions and actual values
    test_pred_unscaled = scaler.inverse_transform(test_pred.reshape(-1, 1))
    test_actual_unscaled = scaler.inverse_transform(test_actual.reshape(-1, 1))

    # print predicted and actual values

    import matplotlib.pyplot as plt

    # plot predicted and actual values
    test_df = df.tail(len(test_actual_unscaled))

    plt.plot(test_df.index, test_actual_unscaled, label='actual')
    plt.plot(test_df.index, test_pred_unscaled, label='predicted')
    plt.legend()
    plt.show()
    # get the last timestamp, last close value, and last predicted value
    last_timestamp = test_df.index[-1]
    last_close = test_df['close'][-1]
    last_predicted = test_pred_unscaled[-1][0]

    # print the values
    print(f"Last timestamp: {last_timestamp}")
    print(f"Last close value: {last_close}")
    print(f"Last predicted value: {last_predicted}")

