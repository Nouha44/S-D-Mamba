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
from sklearn.metrics import mean_squared_error, mean_absolute_error

from model.S_Mamba import Model  # import your S_Mamba model here

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
    series = series.reshape(-1, series.shape[1])  # [time, features]
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
            X.append(data[i:i+seq_len].T)  # [features, seq_len]
            Y.append(data[i+seq_len:i+seq_len+pred_len].T)  # [features, pred_len]
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
def train_model(model, train_loader, val_loader, device, lr=1e-4, epochs=50, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []
        time_start = time.time()
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.float().permute(0, 2, 1).to(device)  # [B, N, L]
            Y_batch = Y_batch.float().permute(0, 2, 1).to(device)

            optimizer.zero_grad()
            # decoder input: label_len past + zeros for pred_len
            label_len = min(48, Y_batch.shape[2])
            dec_inp = torch.cat([Y_batch[:, :, :label_len],
                                 torch.zeros(Y_batch.shape[0], Y_batch.shape[1], Y_batch.shape[2]-label_len).to(device)],
                                dim=2)
            output = model(X_batch, None, dec_inp, None)
            output = output[:, :, -Y_batch.shape[2]:]
            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val = X_val.float().permute(0, 2, 1).to(device)
                Y_val = Y_val.float().permute(0, 2, 1).to(device)
                dec_inp = torch.cat([Y_val[:, :, :label_len],
                                     torch.zeros(Y_val.shape[0], Y_val.shape[1], Y_val.shape[2]-label_len).to(device)],
                                    dim=2)
                output = model(X_val, None, dec_inp, None)[:, :, -Y_val.shape[2]:]
                val_losses.append(criterion(output, Y_val).item())

        val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {np.mean(train_losses):.6f} | Val Loss: {val_loss:.6f} | Time: {time.time()-time_start:.1f}s")

        early_stopping(val_loss, model, "./checkpoints/weather")
        if early_stopping.early_stop:
            print("⏹ Early stopping")
            break

    # Load best model
    model.load_state_dict(torch.load("./checkpoints/weather/checkpoint.pth"))
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
            X = X_batch[0].float().permute(1,0).unsqueeze(0).to(device)
            Y = Y_batch[0].float().permute(1,0).unsqueeze(0).to(device)

            dec_inp = torch.zeros_like(Y).to(device)
            pred = model(X, None, dec_inp, None)[:, :, -Y.shape[2]:]

            # inverse scale
            context = scaler.inverse_transform(X.cpu().numpy().squeeze(-1).T)
            true_future = scaler.inverse_transform(Y.cpu().numpy().squeeze(-1).T)
            forecast = scaler.inverse_transform(pred.cpu().numpy().squeeze(-1).T)

            plt.plot(range(seq_len), context, color="green", label="Context")
            plt.plot(range(seq_len, seq_len+pred_len), true_future, color="gold", label="Ground Truth")
            plt.plot(range(seq_len, seq_len+pred_len), forecast, color="red", label="Forecast")
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
    series = df.values[:, 1:]  # assume first column is date/time, rest are features

    seq_len = 96
    label_len = 48
    pred_len = 96
    batch_size = 32

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), scaler = prepare_datasets(
        series, seq_len, label_len, pred_len
    )

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(Y_val)),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(Y_test)),
                             batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create S_Mamba model
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
