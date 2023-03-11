import torch
from matplotlib.pyplot import plot, grid, legend, show, title
from datetime import datetime
from os import path
from numpy import concatenate, array, float32
from pytorch_lightning import callbacks, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelSummary
from models.lstm_model import LSTMBidirectional


class LSTMTrainer:
    def __init__(self, hidden_size=64, num_layers=2, output_size=1, learning_rate=0.0001, weight_decay=1e-3,
                 dropout=0.2, sequence_length=24, batch_size=128, num_epochs=200):
        # Constructor to initialize models variables
        self.dir_path = "save"
        self.input_size = None
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.scaler = MinMaxScaler()
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
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.checkpoint_callback = None
        self.early_stopping_callback = None
        self.model_summary_callback = None
        self.logger = None
        self.trainer = None

    def preprocess_data(self, data_file):
        self.df = data_file
        self.labels = self.df['close_pct_change'].shift(-1).fillna(0).values.reshape(-1, 1)
        self.features = self.df.drop(columns=['close', 'close_pct_change']).values
        self.features_scaled = self.scaler.fit_transform(self.features)
        self.labels_scaled = self.scaler.fit_transform(self.labels)
        # Convert the data into inputs and outputs
        for i in range(self.sequence_length, len(self.features_scaled)):
            self.inputs.append(self.features_scaled[i - self.sequence_length:i])
            self.outputs.append(self.labels_scaled[i])
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

        self.model = LSTMBidirectional(self.features.shape[1], self.hidden_size, self.num_layers, self.output_size,
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
            #save_last=True
        )
        # Configure early stopping and summary callback
        self.early_stopping_callback = EarlyStopping(monitor="val_loss", patience=15, mode="min")
        # Configure TensorBoard Logger
        self.logger = TensorBoardLogger(save_dir='./lightning_logs', name='bilstm-regressor', default_hp_metric=True)
        self.logger.log_hyperparams({
            "input_size": self.features.shape[1],
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
                               callbacks=[self.checkpoint_callback, self.early_stopping_callback],
                               auto_lr_find=auto_lr_find,
                               auto_scale_batch_size=auto_scale_batch_size)
        self.model.to(self.device)
        self.trainer.fit(self.model, self.train_loader, self.val_loader)

    def save_model(self):
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        file_name = f"best_model_{self.features.shape[1]}_{self.hidden_size}_{self.num_layers}_{self.dropout}_{time_stamp}.pt"
        file_path = path.join(self.dir_path, file_name)
        torch.save(self.model.state_dict(), file_path)

    def test_model(self, ckpt_path="best"):
        self.model.to(self.device)
        self.trainer.test(self.model, self.test_loader, ckpt_path=ckpt_path)

    def evaluate_model(self):
        # Evaluate model's performance on the test set
        self.model.to(self.device)
        self.model.eval()
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
        plot(test_df.index, test_actual_unscaled, label='actual')
        plot(test_df.index, test_pred_unscaled, label='predicted')
        #plot(test_df.index, test_df['close'], label='close')
        title(
            f"best_model_{self.features.shape[1]}{self.hidden_size}{self.num_layers}{self.dropout}{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        grid(True)
        legend()
        show()
        # Print last timestamp, actual close value, and predicted close value
        last_timestamp = test_df.index[-1]
        last_close_percentage_change = test_df.loc[test_df.index[-1], 'close_pct_change']
        last_close_value = test_df.loc[test_df.index[-1], 'close']
        last_predicted = test_pred_unscaled[-1][0]
        print(f"Last timestamp: {last_timestamp}")
        print(f"Last close change value: {last_close_percentage_change}")
        print(f"Last close value: {last_close_value}")
        print(f"Last predicted value: {last_predicted}")