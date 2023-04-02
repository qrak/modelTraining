import torch
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from os import path
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from models.lstm_model import BitcoinPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_scaler = MinMaxScaler()
        self.label_scaler = MinMaxScaler()
        self.checkpoint_callback = callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints/',
            filename='best_model_{epoch}_{val_loss:.4f}',
            save_top_k=1,
            mode='min'
        )
        self.early_stopping_callback = EarlyStopping(monitor="val_loss", patience=15, mode="min")
        self.logger = TensorBoardLogger(save_dir='./lightning_logs', name='bilstm-regressor', default_hp_metric=True)
        self.auto_lr_find = True
        self.auto_scale_batch_size = True
        self.trainer = Trainer(max_epochs=self.config['num_epochs'],
                               accelerator="gpu" if torch.cuda.is_available() else 0,
                               logger=self.logger, log_every_n_steps=1,
                               callbacks=[self.checkpoint_callback, self.early_stopping_callback],
                               auto_lr_find=self.auto_lr_find,
                               auto_scale_batch_size=self.auto_scale_batch_size)
        self.config['save_dir'] = "save"
        self.lr_scheduler = None
        self.optimizer = None
        self.optimizer_config = None
        self.model = None
        self.test_loader = None
        self.val_loader = None
        self.train_loader = None
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.y_val = None
        self.y_train = None
        self.X_val = None
        self.X_train = None
        self.y_test = None
        self.y_train_val = None
        self.X_test = None
        self.X_train_val = None
        self.outputs_tensor = None
        self.inputs_tensor = None
        self.outputs_array = None
        self.inputs_array = None
        self.labels_scaled = None
        self.features_scaled = None
        self.labels = None
        self.features = None
        self.df = None
        self.model_state_dict = None
    @staticmethod
    def preprocess_chunk(chunk):
        if 'timestamp' in chunk.columns:
            chunk.set_index('timestamp', inplace=True)
            chunk.sort_index(inplace=True)
        return chunk

    @staticmethod
    def create_sequences(data, seq_length, is_label=False):
        data_len, num_features = data.shape
        num_sequences = data_len - seq_length

        if is_label:
            sequences = data[seq_length:]
        else:
            sequences = np.zeros((num_sequences, seq_length, num_features))
            for i in tqdm(range(seq_length), desc="Creating sequences"):
                sequences[:, i, :] = data[i:num_sequences + i, :]

        return sequences

    def preprocess_data(self, data, chunksize=None, input_type='file'):
        features_list = []
        labels_list = []
        dfs_list = []
        try:
            if input_type == 'file':
                reader = pd.read_csv(data, chunksize=chunksize, iterator=True)
                for chunk in tqdm(reader, desc="Processing chunks"):
                    chunk = self.preprocess_chunk(chunk)
                    dfs_list.append(chunk)
            elif input_type == 'dataframe':
                if chunksize:
                    dfs_list = []
                    for i in range(0, len(data), chunksize):
                        chunk = data[i:i + chunksize]
                        chunk = self.preprocess_chunk(chunk)
                        dfs_list.append(chunk)
                else:
                    dfs_list = [self.preprocess_chunk(data)]
            else:
                raise ValueError("Invalid input_type. Must be either 'file' or 'dataframe'.")
        except Exception as e:
            logger.error(f"Error in preprocess_data: {e}")

        for chunk in dfs_list:
            features = chunk.drop(
                columns=['close', 'timestamp']).values if 'timestamp' in chunk.columns else chunk.drop(
                columns=['close']).values
            labels = chunk['close'].values.reshape(-1, 1)
            features = self.create_sequences(features, self.config['sequence_length'])
            labels = self.create_sequences(labels, self.config['sequence_length'], is_label=True)
            labels = labels.reshape(-1, 1)

            features_scaled = self.feature_scaler.fit_transform(
                features.reshape(-1, features.shape[-1])).reshape(features.shape)
            labels_scaled = self.label_scaler.fit_transform(labels)
            features_list.append(np.array(features_scaled, dtype=np.float32))
            labels_list.append(np.array(labels_scaled, dtype=np.float32))

        self.df = pd.concat(dfs_list)  # Concatenate all chunks into a single DataFrame and assign it to self.df
        self.inputs_array = np.concatenate(features_list, axis=0)
        self.outputs_array = np.concatenate(labels_list, axis=0)
        self.features = self.inputs_array
        self.inputs_tensor = torch.tensor(self.inputs_array, dtype=torch.float32)
        self.outputs_tensor = torch.tensor(self.outputs_array, dtype=torch.float32)

    def split_data(self, test_size, random_state):
        try:
            self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(self.inputs_tensor,
                                                                                            self.outputs_tensor,
                                                                                            test_size=test_size,
                                                                                            random_state=random_state,
                                                                                            shuffle=True)

            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_val, self.y_train_val,
                                                                                  test_size=0.2,
                                                                                  random_state=random_state + 1,
                                                                                  shuffle=True)
            self.train_dataset = TensorDataset(self.X_train, self.y_train)
            self.val_dataset = TensorDataset(self.X_val, self.y_val)
            self.test_dataset = TensorDataset(self.X_test, self.y_test)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['batch_size'],
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=4)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.config['batch_size'],
                                         shuffle=False,
                                         drop_last=True,
                                         num_workers=4)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.config['batch_size'],
                                          shuffle=False,
                                          drop_last=True)
        except Exception as e:
            logger.error(f"Error in split_data: {e}")

    def configure_model(self):
        try:
            self.config['input_size'] = self.features.shape[-1]
            self.model = BitcoinPredictor(self.config)
            self.logger.log_hyperparams(self.config)
        except Exception as e:
            logger.error(f"Error in configure_model: {e}")

    def train_model(self):
        try:
            self.model.to(self.device)
            self.optimizer_config = self.model.configure_optimizers()
            self.optimizer = self.optimizer_config['optimizer']
            self.lr_scheduler = self.optimizer_config['lr_scheduler']
            self.trainer.fit(self.model, self.train_loader, self.val_loader)
        except Exception as e:
            logger.error(f"Error in train_model: {e}")

    def save_model(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            file_name = f"best_model_{self.config['hidden_size']}_{self.config['num_layers']}_{self.config['dropout']}_{timestamp}.pt"
            file_path = path.join(self.config['save_dir'], file_name)
            torch.save(self.model.state_dict(), file_path)
        except Exception as e:
            logger.error(f"Error in save_model: {e}")

    def load_model(self, file_path):
        try:
            self.model_state_dict = torch.load(file_path, map_location=self.device)
            self.model.load_state_dict(self.model_state_dict)
        except Exception as e:
            logger.error(f"Error in load_model: {e}")

    def test_model(self, ckpt_path="best"):
        try:
            self.model.to(self.device)
            self.trainer.test(self.model, self.test_loader, ckpt_path=ckpt_path)
        except Exception as e:
            logger.error(f"Error in test_model: {e}")

    def evaluate_model(self, tail_n=200):
        try:
            self.model.to(self.device)
            self.model.eval()
            test_pred = []
            test_actual = []
            with torch.no_grad():
                for batch in self.test_loader:
                    x, y = batch
                    x = x.to(self.device)
                    y_pred = self.model(x)
                    test_pred.append(y_pred.cpu().numpy())
                    test_actual.append(y.cpu().numpy())

            test_pred = np.concatenate(test_pred, axis=0)
            test_actual = np.concatenate(test_actual, axis=0)
            test_pred_unscaled = self.label_scaler.inverse_transform(test_pred.reshape(-1, 1))
            test_actual_unscaled = self.label_scaler.inverse_transform(test_actual.reshape(-1, 1))
            # Plot predicted and actual values against time
            test_df = self.df.tail(tail_n)
            test_df.index = pd.to_datetime(test_df.index)

            plt.plot(test_df.index, test_actual_unscaled[-tail_n:], label='actual')
            plt.plot(test_df.index, test_pred_unscaled[-tail_n:], label='predicted')
            # plot(test_df.index, test_df['close'], label='close')
            plt.title(
                f"best_model_{self.features.shape[1]}_{self.config['hidden_size']}_{self.config['num_layers']}_{self.config['dropout']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            plt.grid(True)
            plt.legend()
            plt.show()
            # Print last timestamp, actual close value, and predicted close value
            last_timestamp = test_df.index[-1]
            last_close_value = test_df.loc[test_df.index[-1], 'close']
            last_predicted = test_pred_unscaled[-1][0]
            print(f"Last timestamp: {last_timestamp}")
            print(f"Last close value: {last_close_value}")
            print(f"Last predicted value: {last_predicted}")
            print("All predicted values:", test_pred_unscaled.reshape(-1))
            print("All actual values:", test_actual_unscaled.reshape(-1))
        except Exception as e:
            logger.error(f"Error in evaluate_model: {e}")
