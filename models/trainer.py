from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from models.stock_crypto_predictor import StockCryptoPredictor
from tqdm import tqdm
from datetime import datetime
from os import path
import torch
import pandas as pd
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # or Qt5Agg
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelTrainer:
    """
        Class for training, testing, and evaluating a stock/crypto price prediction model.

        Attributes:
            logger: A logger object for logging messages.
            config: A dictionary containing the configuration for the model.
            device: A string representing the device on which to train the model.
            feature_scaler: A StandardScaler object for scaling the features.
            label_scaler: A StandardScaler object for scaling the labels.
            checkpoint_callback: A ModelCheckpoint object for saving the best model during training.
            early_stopping_callback: An EarlyStopping object for stopping training early if the validation loss does not improve.
            tensorboard_logger: A TensorBoardLogger object for logging training progress.
            trainer: A Trainer object for training the model.
            lr_scheduler: An object for adjusting the learning rate during training.
            optimizer: An optimizer object for optimizing the model's parameters.
            optimizer_config: A dictionary containing the configuration for the optimizer.
            model: A StockCryptoPredictor object representing the model.
            test_loader: A DataLoader object for loading the test data.
            val_loader: A DataLoader object for loading the validation data.
            train_loader: A DataLoader object for loading the training data.
            test_dataset: A TensorDataset object for holding the test data.
            val_dataset: A TensorDataset object for holding the validation data.
            train_dataset: A TensorDataset object for holding the training data.
            y_val: A numpy array representing the labels of the validation data.
            y_train: A numpy array representing the labels of the training data.
            X_val: A numpy array representing the features of the validation data.
            X_train: A numpy array representing the features of the training data.
            y_test: A numpy array representing the labels of the test data.
            y_train_val: A numpy array representing the labels of the training and validation data.
            X_test: A numpy array representing the features of the test data.
            X_train_val: A numpy array representing the features of the training and validation data.
            outputs_tensor: A tensor representing the labels of the data.
            inputs_tensor: A tensor representing the features of the data.
            outputs_array: A numpy array representing the labels of the data.
            inputs_array: A numpy array representing the features of the data.
            labels_scaled: A numpy array representing the scaled labels of the data.
            features_scaled: A numpy array representing the scaled features of the data.
            labels: A numpy array representing the labels of the data.
            features: A numpy array representing the features of the data.
            df: A pandas DataFrame representing the concatenated chunks of the input data.
            model_state_dict: A dictionary representing the state of the model.

        Methods:
            preprocess_chunk(chunk): Preprocesses a single chunk of data.
            create_sequences(data, seq_length, is_label=False): Creates sequences from the data.
            preprocess_data(data, chunksize=None, input_type='file'): Preprocesses the data.
            split_data(test_size, random_state): Splits the data into training, validation, and test sets.
            configure_model(): Configures the model.
            train_model(): Trains the model.
            save_model(): Saves the model.
            load_model(file_path): Loads the model from the specified file.
            test_model(ckpt_path="best"): Tests the model on the test set.
            evaluate_model(tail_n=200): Evaluates the model by plotting the predicted and actual values against time.
        """
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_scaler = StandardScaler()
        self.label_scaler = StandardScaler()
        self.checkpoint_callback = callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints/',
            filename='best_model_{epoch}_{val_loss:.4f}',
            save_top_k=1,
            mode='min'
        )
        self.early_stopping_callback = EarlyStopping(monitor="val_loss", patience=15, mode="min")
        self.tensorboard_logger = None
        self.trainer = None
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
            self.logger.error(f"Error in preprocess_data: {e}")

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
            self.logger.error(f"Error in split_data: {e}")

    def configure_model(self):
        try:
            self.config['input_size'] = self.features.shape[-1]
            self.model = StockCryptoPredictor(self.config)
        except Exception as e:
            self.logger.error(f"Error in configure_model: {e}")

    def train_model(self):
        try:
            self.tensorboard_logger = TensorBoardLogger(save_dir='./lightning_logs', name='crypto-prediction',
                                                        default_hp_metric=True)
            self.tensorboard_logger.log_hyperparams(self.config)
            self.trainer = Trainer(max_epochs=self.config['num_epochs'],
                                   accelerator="gpu" if torch.cuda.is_available() else "cpu",
                                   logger=self.tensorboard_logger, log_every_n_steps=1,
                                   callbacks=[self.checkpoint_callback, self.early_stopping_callback])
            self.model.to(self.device)
            self.optimizer_config = self.model.configure_optimizers()
            self.optimizer = self.optimizer_config['optimizer']
            self.lr_scheduler = self.optimizer_config['lr_scheduler']
            # Log the training configuration
            self.logger.info(f"Training the model with the following configuration:\n"
                             f"  hidden_size: {self.config['hidden_size']}\n"
                             f"  num_layers: {self.config['num_layers']}\n"
                             f"  num_heads: {self.config['num_heads']}\n"
                             f"  learning_rate: {self.config['learning_rate']}\n"
                             f"  weight_decay: {self.config['weight_decay']}\n"
                             f"  dropout: {self.config['dropout']}\n"
                             f"  sequence_length: {self.config['sequence_length']}\n"
                             f"  batch_size: {self.config['batch_size']}\n"
                             f"  num_epochs: {self.config['num_epochs']}\n")
            self.trainer.fit(self.model, self.train_loader, self.val_loader)
            self.logger.info(f"Model training completed!")
        except Exception as e:
            self.logger.error(f"Error in train_model: {e}")

    def save_model(self, best_model_ckpt_path=None):
        try:
            if best_model_ckpt_path:
                checkpoint = torch.load(best_model_ckpt_path)
                self.model.load_state_dict(checkpoint['state_dict'])

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            file_name = f"best_model_{self.config['hidden_size']}_" \
                        f"{self.config['num_layers']}_" \
                        f"{self.config['num_heads']}_" \
                        f"{self.config['learning_rate']}_" \
                        f"{self.config['weight_decay']}_" \
                        f"{self.config['dropout']}_" \
                        f"{self.config['sequence_length']}_" \
                        f"{self.config['batch_size']}_" \
                        f"{self.config['num_epochs']}_" \
                        f"{timestamp}.pt"
            file_path = path.join(self.config['save_dir'], file_name)
            self.logger.info(f"Saving model...")
            torch.save(self.model.state_dict(), file_path)
            self.logger.info(f"Model saved successfully.")
        except Exception as e:
            self.logger.error(f"Error in save_model: {e}")

    def load_model(self, file_path):
        try:
            self.logger.info(f"Loading the model...")
            self.model_state_dict = torch.load(file_path, map_location=self.device)
            self.model.load_state_dict(self.model_state_dict)
            self.logger.info(f"Model loaded.")
        except Exception as e:
            self.logger.error(f"Error in load_model: {e}")

    def test_model(self, ckpt_path="best"):
        try:
            self.logger.info(f"Testing the model on test set...")
            self.model.to(self.device)
            self.trainer.test(self.model, self.test_loader, ckpt_path=ckpt_path)
            self.logger.info(f"Model testing on test set finished.")
        except Exception as e:
            self.logger.error(f"Error in test_model: {e}")

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
            if tail_n:
                test_df = self.df.tail(tail_n)
            else:
                test_df = self.df
            test_df.index = pd.to_datetime(test_df.index)

            # Add past predictions to the test DataFrame
            past_predictions = pd.DataFrame(test_pred_unscaled[:tail_n], index=test_df.index, columns=['predicted'])
            test_df = pd.concat([test_df, past_predictions], axis=1)

            # Add future predictions to the test DataFrame
            future_index = pd.date_range(test_df.index[-1] + pd.Timedelta(minutes=1),
                                         periods=self.config['sequence_length'], freq='T')
            future_predictions = pd.DataFrame(test_pred_unscaled[-self.config['sequence_length']:], index=future_index,
                                              columns=['predicted'])
            test_df = pd.concat([test_df, future_predictions])

            # Print last timestamp, actual close value, and predicted close value
            last_timestamp = test_df.index[-1]
            last_close_value = self.df.loc[self.df.index[-1], 'close']
            last_predicted = test_pred_unscaled[-1][0]
            self.logger.info(f"Last timestamp: {last_timestamp}")
            self.logger.info(f"Last close value: {last_close_value}")
            self.logger.info(f"Last predicted value: {last_predicted}")

            plt.plot(test_df.index[:tail_n], test_actual_unscaled[-tail_n:], label='actual')
            plt.plot(test_df.index, test_df['predicted'], label='predicted')
            plt.title(f"loaded model: hidden_size: {self.config['hidden_size']} "
                      f"num_layers: {self.config['num_layers']} "
                      f"num_heads: {self.config['num_heads']} "
                      f"lr: {self.config['learning_rate']} "
                      f"weight_decay: {self.config['weight_decay']} "
                      f"dropout: {self.config['dropout']} "
                      f"sequence_length: {self.config['sequence_length']} "
                      f"batch_size: {self.config['batch_size']} "
                      f"num_epochs: {self.config['num_epochs']} ")
            plt.grid(True)
            plt.legend()
            plt.show()
        except Exception as e:
            self.logger.error(f"Error in evaluate_model: {e}")

# for future use
# self.logger.info("All predicted values: %s", test_pred_unscaled.reshape(-1))
# self.logger.info("All actual values: %s", test_actual_unscaled.reshape(-1))
