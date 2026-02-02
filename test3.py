import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from model.MSM import Model  # your updated MSM model

# =====================================================
# CONFIG
# =====================================================
CSV_PATH = "./dataset/weather/weather.csv"
SEQ_LEN = 128
PRED_LEN = 128
LABEL_LEN = SEQ_LEN // 2
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "./results"
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# DATASET
# =====================================================
class WeatherDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        x_mark = torch.zeros_like(x)  # placeholder for time features
        y_mark = torch.zeros_like(y)
        return x, x_mark, y, y_mark

# =====================================================
# LOAD + SPLIT + STANDARDIZE
# =====================================================
df = pd.read_csv(CSV_PATH)
values = df["values"].values.astype("float32").reshape(-1, 1)
num_vars = values.shape[1]

n = len(values)
train_end = int(0.7 * n)
val_end = int(0.8 * n)

scaler = StandardScaler()
train_data = scaler.fit_transform(values[:train_end])
val_data   = scaler.transform(values[train_end:val_end])
test_data  = scaler.transform(values[val_end:])

train_ds = WeatherDataset(train_data, SEQ_LEN, PRED_LEN)
val_ds   = WeatherDataset(val_data, SEQ_LEN, PRED_LEN)
test_ds  = WeatherDataset(test_data, SEQ_LEN, PRED_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# =====================================================
# MODEL CONFIG
# =====================================================
class Config:
    seq_len = SEQ_LEN
    pred_len = PRED_LEN
    d_model = 512
    d_state = 2
    d_ff = 512
    e_layers = 3
    dropout = 0.1
    activation = "gelu"
    embed = "timeF"
    freq = "h"
    output_attention = False
    use_norm = True
    class_strategy = "projection"

model = Model(Config(), num_vars=num_vars).to(DEVICE)

# =====================================================
# TRAINING LOOP
# =====================================================
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
best_val = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for x, x_mark, y, y_mark in train_loader:
        x, x_mark = x.to(DEVICE), x_mark.to(DEVICE)
        y, y_mark = y.to(DEVICE), y_mark.to(DEVICE)

        # Decoder input: first half of true y + zeros
        dec_inp = torch.cat([y[:, :LABEL_LEN, :], torch.zeros_like(y[:, LABEL_LEN:, :])], dim=1)

        optimizer.zero_grad()
        preds = model(x, x_mark, dec_inp, y_mark)
        preds = preds[:, -PRED_LEN:, :]
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, x_mark, y, y_mark in val_loader:
            x, x_mark = x.to(DEVICE), x_mark.to(DEVICE)
            y, y_mark = y.to(DEVICE), y_mark.to(DEVICE)
            dec_inp = torch.cat([y[:, :LABEL_LEN, :], torch.zeros_like(y[:, LABEL_LEN:, :])], dim=1)
            preds = model(x, x_mark, dec_inp, y_mark)
            preds = preds[:, -PRED_LEN:, :]
            val_loss += criterion(preds, y).item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1:02d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

    # Save best model
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_msm_weather.pt"))

# =====================================================
# FORECAST + RMSE
# =====================================================
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_msm_weather.pt")))
model.eval()

def evaluate_rmse(model, loader, scaler, device):
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for x, x_mark, y, y_mark in loader:
            x, x_mark = x.to(device), x_mark.to(device)
            y, y_mark = y.to(device), y_mark.to(device)
            dec_inp = torch.cat([y[:, :LABEL_LEN, :], torch.zeros_like(y[:, LABEL_LEN:, :])], dim=1)
            preds = model(x, x_mark, dec_inp, y_mark)
            preds = preds[:, -PRED_LEN:, :]

            # Inverse scaling
            preds_real = scaler.inverse_transform(preds.cpu().numpy().reshape(-1, 1))
            y_real = scaler.inverse_transform(y.cpu().numpy().reshape(-1, 1))

            all_preds.append(preds_real)
            all_trues.append(y_real)

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    mse = np.mean((all_preds - all_trues) ** 2)
    rmse = np.sqrt(mse)
    return rmse

test_rmse = evaluate_rmse(model, test_loader, scaler, DEVICE)
print(f"📊 Test RMSE (after inverse transform): {test_rmse:.6f}")

# =====================================================
# PLOT FIRST PREDICTION
# =====================================================
x, x_mark, y, y_mark = next(iter(test_loader))
x, x_mark = x.to(DEVICE), x_mark.to(DEVICE)
y, y_mark = y.to(DEVICE), y_mark.to(DEVICE)
dec_inp = torch.cat([y[:, :LABEL_LEN, :], torch.zeros_like(y[:, LABEL_LEN:, :])], dim=1)

with torch.no_grad():
    preds = model(x, x_mark, dec_inp, y_mark)
    preds = preds[:, -PRED_LEN:, :]

context = scaler.inverse_transform(x[0].cpu().numpy())
true_future = scaler.inverse_transform(y[0].cpu().numpy())
forecast = scaler.inverse_transform(preds[0].cpu().numpy())

plt.figure(figsize=(14, 5))
plt.plot(range(SEQ_LEN), context[:, 0], label="Context", color="green")
plt.plot(range(SEQ_LEN, SEQ_LEN + PRED_LEN), true_future[:, 0], label="Ground Truth", color="gold")
plt.plot(range(SEQ_LEN, SEQ_LEN + PRED_LEN), forecast[:, 0], label="Forecast", color="red")
plt.title("MSM Weather Forecast")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, "weather_forecast.png"), dpi=200)
plt.show()
print(f"✅ Forecast plot saved to {SAVE_DIR}/weather_forecast.png")
