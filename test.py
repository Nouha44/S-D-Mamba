import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from model.S_Mamba import Model  # make sure your S_Mamba.py is correct

# ------------------------------
# 1. Seed
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------
# 2. Dataset preparation
# ------------------------------
def prepare_datasets(series, seq_len, label_len, pred_len, val_ratio=0.2, test_ratio=0.2, normalize=True):
    """
    series: [time, features]
    Returns: (X_train,Y_train), (X_val,Y_val), (X_test,Y_test), scaler
    All windows: [samples, features, seq_len/pred_len]
    """
    series = series.astype(np.float32)
    n = len(series)

    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))

    train = series[:train_end]
    val = series[train_end - seq_len:val_end]
    test = series[val_end - seq_len:]

    scaler = StandardScaler()
    if normalize:
        scaler.fit(train)
        train = scaler.transform(train)
        val = scaler.transform(val)
        test = scaler.transform(test)

    def create_windows(data):
        X, Y = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            x_win = data[i:i+seq_len].T  # [features, seq_len]
            y_win = data[i+seq_len:i+seq_len+pred_len].T  # [features, pred_len]
            X.append(x_win)
            Y.append(y_win)
        return np.array(X), np.array(Y)

    return (
        create_windows(train),
        create_windows(val),
        create_windows(test),
        scaler
    )

# ------------------------------
# 3. Early Stopping
# ------------------------------
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# ------------------------------
# 4. Training function
# ------------------------------
def train_model(model, train_loader, val_loader, device, lr=5e-5, epochs=50, patience=5, checkpoint_path="./checkpoints/weather"):
    os.makedirs(checkpoint_path, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        t0 = time.time()
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.float().to(device)  # [B, features, seq_len]
            Y_batch = Y_batch.float().to(device)  # [B, features, pred_len]

            optimizer.zero_grad()

            label_len = min(48, Y_batch.shape[2])
            dec_inp = torch.cat([Y_batch[:, :, :label_len],
                                 torch.zeros(Y_batch.shape[0], Y_batch.shape[1], Y_batch.shape[2]-label_len).to(device)],
                                dim=2)

            output = model(X_batch.unsqueeze(0), None, dec_inp.unsqueeze(0), None)  # [1,B,features,pred_len] ?
            output = output[:, :, :, -Y_batch.shape[2]:]  # ensure only pred_len

            loss = criterion(output.squeeze(0), Y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val = X_val.float().to(device)
                Y_val = Y_val.float().to(device)
                dec_inp = torch.cat([Y_val[:, :, :label_len],
                                     torch.zeros(Y_val.shape[0], Y_val.shape[1], Y_val.shape[2]-label_len).to(device)],
                                    dim=2)
                output = model(X_val.unsqueeze(0), None, dec_inp.unsqueeze(0), None)[:, :, :, -Y_val.shape[2]:]
                val_losses.append(criterion(output.squeeze(0), Y_val).item())

        val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {np.mean(train_losses):.6f} | Val Loss: {val_loss:.6f} | Time: {time.time()-t0:.1f}s")
        early_stopping(val_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            print("⏹ Early stopping")
            break

    # Load best model
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'checkpoint.pth')))
    return model

# ------------------------------
# 5. Forecast and plot
# ------------------------------
def forecast_and_plot(model, data_loader, scaler, device, seq_len, pred_len, save_path="./results/weather_forecast.png"):
    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(15,5))

    with torch.no_grad():
        for X_batch, Y_batch in data_loader:
            X = X_batch[0].float().to(device)  # [features, seq_len]
            Y = Y_batch[0].float().to(device)  # [features, pred_len]

            dec_inp = torch.zeros_like(Y).to(device)
            pred = model(X.unsqueeze(0), None, dec_inp.unsqueeze(0), None)[:, :, :, -Y.shape[1]:]

            # convert to [time, features] for scaler
            context = X.cpu().numpy().T
            true_future = Y.cpu().numpy().T
            forecast = pred[0].cpu().numpy().T

            context = scaler.inverse_transform(context)
            true_future = scaler.inverse_transform(true_future)
            forecast = scaler.inverse_transform(forecast)

            # Plot first feature
            plt.plot(range(seq_len), context[:,0], color="green", label="Context")
            plt.plot(range(seq_len, seq_len+pred_len), true_future[:,0], color="gold", label="Ground Truth")
            plt.plot(range(seq_len, seq_len+pred_len), forecast[:,0], color="red", label="Forecast")
            break

    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title("Weather Forecast vs Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

# ------------------------------
# 6. Main
# ------------------------------
if __name__ == "__main__":
    set_seed(42)
    dataset_path = "./dataset/weather/weather.csv"
    df = pd.read_csv(dataset_path)
    series = df.values[:, 1:]  # skip date column

    seq_len = 96
    label_len = 48
    pred_len = 96
    batch_size = 32

    (X_train,Y_train), (X_val,Y_val), (X_test,Y_test), scaler = prepare_datasets(
        series, seq_len, label_len, pred_len
    )

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(Y_val)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(Y_test)), batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class Config:
        seq_len = seq_len
        pred_len = pred_len
        d_model = 512
        d_ff = 512
        e_layers = 3
        output_attention = False
        use_norm = True
        class_strategy = "projection"
        embed = "timeF"
        freq = "h"
        dropout = 0.1
        d_state = 2
        activation = "gelu"

    configs = Config()
    model = Model(configs).to(device)

    os.makedirs("./checkpoints/weather", exist_ok=True)
    model = train_model(model, train_loader, val_loader, device,
                        lr=5e-5, epochs=5, patience=3)

    os.makedirs("./results", exist_ok=True)
    forecast_and_plot(model, test_loader, scaler, device, seq_len, pred_len)
