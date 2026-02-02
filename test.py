import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from model.S_Mamba import Model  # import your S_Mamba model here
from torch.utils.data import DataLoader, TensorDataset

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
def prepare_datasets(series, context_len, pred_len, val_ratio=0.2, test_ratio=0.2, normalize=True):
    series = series.reshape(-1, 1)  # [time, features]
    n = len(series)

    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))

    train = series[:train_end]
    val = series[train_end - context_len:val_end]
    test = series[val_end - context_len:]

    scaler = StandardScaler()
    if normalize:
        scaler.fit(train)
        train = scaler.transform(train)
        val = scaler.transform(val)
        test = scaler.transform(test)

    def create_windows(data):
        X, Y = [], []
        for i in range(len(data) - context_len - pred_len + 1):
            X.append(data[i:i+context_len].T)  # [features, context_len]
            Y.append(data[i+context_len:i+context_len+pred_len].T)  # [features, pred_len]
        return np.array(X), np.array(Y)

    return (
        create_windows(train),
        create_windows(val),
        create_windows(test),
        scaler
    )

# ------------------------------
# 3. Training function
# ------------------------------
def train_model(model, train_loader, val_loader, device, lr=1e-4, epochs=50, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader:
            # Permute to [B, N, L] for S_Mamba
            X_batch = X_batch.float().permute(0, 2, 1).to(device)
            Y_batch = Y_batch.float().permute(0, 2, 1).to(device)

            optimizer.zero_grad()
            # decoder input: zeros with same shape as target
            dec_inp = torch.zeros_like(Y_batch).to(device)
            output = model(X_batch, None, dec_inp, None)
            output = output[:, :, -model.pred_len:]  # [B, N, pred_len]
            loss = criterion(output, Y_batch[:, :, -model.pred_len:])
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
                dec_inp = torch.zeros_like(Y_val).to(device)
                output = model(X_val, None, dec_inp, None)
                output = output[:, :, -model.pred_len:]
                loss = criterion(output, Y_val[:, :, -model.pred_len:])
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {np.mean(train_losses):.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print("⏹ Early stopping")
            break

    # Load best model
    model.load_state_dict(best_state)
    return model

# ------------------------------
# 4. Forecasting and visualization
# ------------------------------
def forecast_and_plot(model, data_loader, scaler, device, context_len, pred_len, save_path="forecast.png"):
    model.eval()
    plt.figure(figsize=(15,5))

    with torch.no_grad():
        for i, (X_batch, Y_batch) in enumerate(data_loader):
            # On ne prend que le premier batch et la première série pour la visualisation
            X = X_batch[0].float().permute(1,0).unsqueeze(0).to(device)  # [1, N, L]
            Y = Y_batch[0].float().permute(1,0).unsqueeze(0).to(device)  # [1, N, L]

            dec_inp = torch.zeros_like(Y).to(device)
            pred = model(X, None, dec_inp, None)[:, :, -pred_len:]  # [1, N, pred_len]

            # Transpose pour avoir [time, features] et inverse scale
            context = X.cpu().numpy().squeeze(-1).T  # [context_len, 1]
            true_future = Y.cpu().numpy().squeeze(-1).T  # [pred_len, 1]
            forecast = pred.cpu().numpy().squeeze(-1).T  # [pred_len, 1]

            # Inverse scaling
            context = scaler.inverse_transform(context)
            true_future = scaler.inverse_transform(true_future)
            forecast = scaler.inverse_transform(forecast)

            # Plot
            plt.plot(range(context_len), context, color="green", label="Context" if i==0 else "")
            plt.plot(range(context_len, context_len+pred_len), true_future, color="gold", label="Ground Truth" if i==0 else "")
            plt.plot(range(context_len, context_len+pred_len), forecast, color="red", label="Forecast" if i==0 else "")

            # On fait juste le premier batch
            break

    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title("Forecast vs Ground Truth with Context Window")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()



# ------------------------------
# 5. Main
# ------------------------------
if __name__ == "__main__":
    set_seed(42)

    dataset_path = "./dataset/weather/weather.csv"
    df = pd.read_csv(dataset_path)
    series = df["OT"].values  # target column

    context_len = 96
    pred_len = 96
    batch_size = 32

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), scaler = prepare_datasets(series, context_len, pred_len)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(Y_val)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(Y_test)), batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    class Config:
        seq_len = context_len
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

    # Train model
    model = train_model(model, train_loader, val_loader, device, lr=5e-5, epochs=5, patience=3)

    # Forecast and plot
    os.makedirs("./results", exist_ok=True)
    forecast_and_plot(model, test_loader, scaler, device, context_len=context_len, pred_len=pred_len, save_path="./results/mamba_forecast_window.png")
