import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from model.S_Mamba import Model  # your S_Mamba model

# -----------------------
# Config (edit here)
# -----------------------
CSV_PATH = "./dataset/weather/weather.csv"
SEQ_LEN = 96
PRED_LEN = 96
BATCH_SIZE = 32
EPOCHS = 20
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Dataset
# -----------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path, seq_len, pred_len):
        df = pd.read_csv(csv_path)

        # drop date column
        values = df.iloc[:, 1:].values.astype("float32")

        self.scaler = StandardScaler()
        values = self.scaler.fit_transform(values)

        self.data = torch.tensor(values)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]

        # time features not used → zeros
        x_mark = torch.zeros_like(x)
        y_mark = torch.zeros_like(y)

        return x, x_mark, y, y_mark

# -----------------------
# Train / Val split
# -----------------------
dataset = TimeSeriesDataset(CSV_PATH, SEQ_LEN, PRED_LEN)

n_train = int(0.7 * len(dataset))
n_val = int(0.1 * len(dataset))
n_test = len(dataset) - n_train - n_val

train_ds, val_ds, test_ds = torch.utils.data.random_split(
    dataset, [n_train, n_val, n_test]
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# -----------------------
# Model config
# -----------------------
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

configs = Config()
model = Model(configs).to(DEVICE)

# -----------------------
# Optimizer & loss
# -----------------------
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# -----------------------
# Training loop
# -----------------------
best_val = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for x, x_mark, y, y_mark in train_loader:
        x, x_mark, y = x.to(DEVICE), x_mark.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        preds = model(x, x_mark, None, None)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, x_mark, y, y_mark in val_loader:
            x, x_mark, y = x.to(DEVICE), x_mark.to(DEVICE), y.to(DEVICE)
            preds = model(x, x_mark, None, None)
            val_loss += criterion(preds, y).item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1:03d} | Train {train_loss:.5f} | Val {val_loss:.5f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_mamba.pt")

# -----------------------
# Test
# -----------------------
model.load_state_dict(torch.load("best_mamba.pt"))
model.eval()

test_loss = 0
with torch.no_grad():
    for x, x_mark, y, y_mark in test_loader:
        x, x_mark, y = x.to(DEVICE), x_mark.to(DEVICE), y.to(DEVICE)
        preds = model(x, x_mark, None, None)
        test_loss += criterion(preds, y).item()

print(f"Test MSE: {test_loss / len(test_loader):.5f}")
