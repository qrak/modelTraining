import torch
import matplotlib.pyplot as plt
from datetime import datetime
from os import path
from numpy import concatenate, array, float32
from pandas import read_csv, to_datetime
from pytorch_lightning import callbacks, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelSummary
from classdirectory.lstm_model import LSTMRegressor


class LSTMTrainer:
    def __init__(self, file_path="save", input_size=10, hidden_size=16, num_layers=2, output_size=1, learning_rate=0.0001, weight_decay=1e-3,
                 dropout=0.2, sequence_length=24, batch_size=128, num_epochs=200):
        # Constructor to initialize class variables
        self.file_path = file_path
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.df = None
        self.features = None
        self.labels = None
        self.features_scaled = None
        self.labels_scaled = None
        self.inputs = []
        self.outputs = []
        self.inputs_array = None
        self.outputs_array = None
        self.inputs_tensor = None
        self.outputs_tensor = None
        self.X_train_val = None
        self.X_test = None
        self.y_train_val = None
        self.y_test = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.scaler = MinMaxScaler()
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.checkpoint_callback = None
        self.early_stopping_callback = None
        self.model_summary_callback = None
        self.logger = None
        self.trainer = None

    def load_data(self, data_file):
        # Load the data from the specified file
        self.df = read_csv(data_file)
        self.df['timestamp'] = to_datetime(self.df['timestamp'], unit='ms')
        self.df.set_index('timestamp', inplace=True)
        self.df = self.df.sort_index()
        # Add new features to the dataframe
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['day_of_month'] = self.df.index.day
        self.df['day_of_year'] = self.df.index.dayofyear
        # Scale the features and labels using MinMaxScaler
        self.features = self.df.drop(columns=['close']).values
        self.labels = self.df['close'].values.reshape(-1, 1)
        self.features_scaled = self.scaler.fit_transform(self.features)
        self.labels_scaled = self.scaler.fit_transform(self.labels)
        # Convert the data into inputs and outputs
        for i in range(len(self.features_scaled) - self.sequence_length):
            self.inputs.append(self.features_scaled[i:i + self.sequence_length])
            self.outputs.append(self.labels_scaled[i + self.sequence_length])
        self.inputs_array = array(self.inputs, dtype=float32)
        self.outputs_array = array(self.outputs, dtype=float32)
        self.inputs_tensor = torch.tensor(self.inputs_array, dtype=torch.float32)
        self.outputs_tensor = torch.tensor(self.outputs_array, dtype=torch.float32)

    def split_data(self, test_size=0.1, random_state=42):
        # Split data into train/validation and test sets
        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(self.inputs_tensor,
                                                                                        self.outputs_tensor,
                                                                                        test_size=test_size,
                                                                                        random_state=random_state,
                                                                                        shuffle=True)
        # Split train/validation set into train and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_val, self.y_train_val,
                                                                              test_size=0.2, random_state=42,
                                                                              shuffle=True)
        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.val_dataset = TensorDataset(self.X_val, self.y_val)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                       num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,
                                     num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def configure_model(self):
        # Configure LSTMRegressor model
        self.model = LSTMRegressor(self.features.shape[1], self.hidden_size, self.num_layers, self.output_size,
                                   self.learning_rate, dropout=self.dropout, weight_decay=self.weight_decay).to(
            self.device)
        # Configure model optimizers
        optimizer_config = self.model.configure_optimizers()
        self.optimizer = optimizer_config['optimizer']
        self.lr_scheduler = optimizer_config['lr_scheduler']
        # Configure checkpoint callback
        self.checkpoint_callback = callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints/',
            filename='best_model_{epoch}_{val_loss:.4f}',
            save_top_k=1,
            mode='min',
            save_last=True
        )
        # Configure early stopping and summary callback
        self.early_stopping_callback = EarlyStopping(monitor="val_loss", patience=15, mode="min")
        self.model_summary_callback = ModelSummary(max_depth=1)
        # Configure TensorBoard Logger
        self.logger = TensorBoardLogger(save_dir='./lightning_logs', name='bilstm-regressor', default_hp_metric=True)
        self.logger.log_hyperparams({
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout
        })

    def train_model(self, auto_lr_find=True, auto_scale_batch_size=True):
        # Train the model
        self.trainer = Trainer(max_epochs=self.num_epochs,
                               accelerator="gpu" if torch.cuda.is_available() else 0,
                               logger=self.logger, log_every_n_steps=1,
                               callbacks=[self.checkpoint_callback, self.early_stopping_callback,
                                          self.model_summary_callback],
                               auto_lr_find=auto_lr_find,
                               auto_scale_batch_size=auto_scale_batch_size)
        self.model.to(self.device)
        self.trainer.fit(self.model, self.train_loader, self.val_loader)

    def save_model(self):
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        file_name = f"best_model_{self.input_size}_{self.hidden_size}_{self.num_layers}_{self.dropout}_{time_stamp}.pt"
        file_path = path.join(self.file_path, file_name)
        torch.save(self.model.state_dict(), file_path)

    def test_model(self, ckpt_path="best"):
        self.model.to(self.device)
        self.trainer.test(self.model, self.test_loader, ckpt_path=ckpt_path)

    def evaluate_model(self):
        # Evaluate model's performance on the test set
        self.model.eval()
        self.model.to(self.device)
        test_pred = []
        test_actual = []
        with torch.no_grad():
            for batch in self.test_loader:
                x, y = batch
                x = x.to(self.device)
                y_pred = self.model(x)

        # Reshape and unscale predicted and actual values
        test_pred.append(y_pred.cpu().numpy())
        test_actual.append(y.cpu().numpy())
        test_pred = concatenate(test_pred, axis=0)
        test_actual = concatenate(test_actual, axis=0)
        test_pred_unscaled = self.scaler.inverse_transform(test_pred.reshape(-1, 1))
        test_actual_unscaled = self.scaler.inverse_transform(test_actual.reshape(-1, 1))
        # Plot predicted and actual values against time
        test_df = self.df.tail(len(test_actual_unscaled))
        plt.plot(test_df.index, test_actual_unscaled, label='actual')
        plt.plot(test_df.index, test_pred_unscaled, label='predicted')
        plt.title(
            f"best_model_{self.input_size}{self.hidden_size}{self.num_layers}{self.dropout}{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        plt.grid(True)
        plt.legend()
        plt.show()
        # Print last timestamp, actual close value, and predicted close value
        last_timestamp = test_df.index[-1]
        last_close = test_df['close'][-1]
        last_predicted = test_pred_unscaled[-1][0]
        print(f"Last timestamp: {last_timestamp}")
        print(f"Last close value: {last_close}")
        print(f"Last predicted value: {last_predicted}")


class LoadModel(LSTMTrainer):
    def __init__(self, model_path):
        super().__init__()
        self.model_state_dict = torch.load(model_path, map_location=self.device)

    def predict(self, df):
        # Load the state dictionary into the model
        self.model.load_state_dict(self.model_state_dict)

        # Pass the new data through the model to generate predictions
        self.model.eval()
        self.model.to(self.device)

        # Get the last row of the original dataframe
        last_row = df.iloc[-1]
        # Create a new input sequence using the last `sequence_length` rows from the original dataframe
        last_inputs = self.features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        last_inputs_tensor = torch.tensor(last_inputs, dtype=torch.float32).to(self.device)

        # Make a prediction for the last row
        with torch.no_grad():
            last_prediction = self.model(last_inputs_tensor).cpu().numpy()

        # Invert the scaling to get the actual predicted value
        last_prediction = self.scaler.inverse_transform(last_prediction)[0][0]

        # Print the last actual value and the predicted value
        print(f"Last actual close price: {last_row['close']}")
        print(f"Last predicted close price: {last_prediction}")


if __name__ == '__main__':
    # input_size is the number of features you have in your csv file
    lstm_trainer = LSTMTrainer("save", input_size=10, hidden_size=8, num_layers=2, output_size=1, learning_rate=0.0001,
    dropout=0.2, weight_decay=1e-3, num_epochs=200)
    lstm_trainer.load_data("BTC_USDT_1h_with_indicators_pivots.csv")
    lstm_trainer.split_data()
    lstm_trainer.configure_model()
    lstm_trainer.train_model()
    lstm_trainer.save_model()
    lstm_trainer.test_model()
    lstm_trainer.evaluate_model()
    lstm_model_loader = LoadModel("save")
    # real data fetch not finished here, for real data check main_load.py
    #lstm_model_loader.predict()