import torch
import torch.nn as nn
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        out, _ = self.lstm(x.to(next(self.parameters()).device))
        out = self.dropout1(out)
        out = self.fc1(out[:, -1, :])
        out = self.dropout2(out)
        out = self.fc2(out)
        return out.to(device)



class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False, path='saved_models/checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')

        # create the saved_models directory if it doesn't exist
        os.makedirs('saved_models', exist_ok=True)

        # save the model in the saved_models directory
        torch.save(model.state_dict(), self.path)