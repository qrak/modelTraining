import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from classdirectory.classfile_test import LSTMRegressor

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
features_train_tensor = torch.tensor(X_train, dtype=torch.float32)
labels_train_tensor = torch.tensor(y_train, dtype=torch.float32)
features_val_tensor = torch.tensor(X_val, dtype=torch.float32)
labels_val_tensor = torch.tensor(y_val, dtype=torch.float32)
features_test_tensor = torch.tensor(X_test, dtype=torch.float32)
labels_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# create datasets
train_dataset = TensorDataset(features_train_tensor, labels_train_tensor)
val_dataset = TensorDataset(features_val_tensor, labels_val_tensor)
test_dataset = TensorDataset(features_test_tensor, labels_test_tensor)

# create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, drop_last=True)

# create model
input_size = features.shape[1]
hidden_size = 16
num_layers = 2
output_size = 1
learning_rate = 1e-3
weight_decay = 1e-4
dropout = 0.2
model = LSTMRegressor.load_from_checkpoint(
    checkpoint_path='save/best_model_epoch=26_val_loss=0.0003.ckpt',
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=output_size,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    dropout=dropout
)

print(model.hparams)



# set the model to evaluation mode
model.eval()

# make predictions on test set
test_pred = []
test_actual = []
with torch.no_grad():
    for batch in test_loader:
        x, y = batch
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
print("Predicted Close Values:\n", test_pred_unscaled.flatten())
print("Actual Close Values:\n", test_actual_unscaled.flatten())

