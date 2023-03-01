import os
import torch
import ccxt
import torch.nn as nn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from classdirectory.classfile import LSTMNet
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from torch.utils.data import DataLoader, TensorDataset


if __name__ == '__main__':

    # Load data into dataframe
    data = pd.read_csv('BTC_USDT_5m_with_indicators.csv', parse_dates=['timestamp'])
    # data = pd.read_csv('csv/BTC_USDT_1m_2022-01-01_now_binance.csv', parse_dates=['timestamp'])
    # Drop any rows with missing values, NaN or infinity
    #data = data.dropna()
    #data = data.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    # Extract the features and target variable
    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop(['timestamp', 'close'], axis=1))
    y = scaler.fit_transform(data[['close']]).ravel()
    # Select the best features
    selector = SelectKBest(mutual_info_regression, k=6)
    selector.fit(X, y)

    # Get the selected features and their scores
    selected_features = data.drop(['timestamp', 'close'], axis=1).columns[selector.get_support()]
    feature_scores = selector.scores_[selector.get_support()]

    print("Selected features:")
    for feature, score in zip(selected_features, feature_scores):
        print(f"{feature}: {score:.4f}")

    # Save the selected features to a file
    filename = "selected_features.txt"
    mode = "a" if os.path.exists(filename) else "w"
    with open(filename, mode) as f:
        if mode == "w":
            f.write("Selected features:\n")
        for feature, score in zip(selected_features, feature_scores):
            f.write(f"{feature}: {score:.4f}\n")
    # Use only the selected features for training and validation
    X = selector.transform(X)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)

    # Scale the target variable separately
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    print("Mean of scaled data:", np.mean(X, axis=0))
    print("Std of scaled data:", np.std(X, axis=0))