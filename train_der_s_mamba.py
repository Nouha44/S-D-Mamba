import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

from model.S_Mamba import Model
from der_continual_s_mamba import DERContinualSMamba


# ---------------- CONFIG ----------------
SEQ_LEN = 128
PRED_LEN = 128
LABEL_LEN = SEQ_LEN // 2
BATCH_SIZE = 64
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- DATASET ----------------
class WeatherDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return x, torch.zeros_like(x), y, torch.zeros_like(y)


def load_task(path):
    series = pd.read_csv(path)["values"].values.astype("float32")
    ds = WeatherDataset(series.reshape(-1, 1), SEQ_LEN, PRED_LEN)
    split = int(0.8 * len(ds))
    return (
        DataLoader(torch.utils.data.Subset(ds, range(split)),
                   batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(torch.utils.data.Subset(ds, range(split, len(ds))),
                   batch_size=BATCH_SIZE)
    )


# ---------------- MAIN ----------------
def main():

    tasks_paths = [
        "/home/nkaraoul/timesfm_backup/mult_sin_d1_full.csv",
        "/home/nkaraoul/timesfm_backup/mult_sin_d2_full.csv",
        "/home/nkaraoul/timesfm_backup/mult_sin_d3_full.csv",
        "/home/nkaraoul/timesfm_backup/mult_sin_d4_full.csv",
    ]

    train_loaders, test_loaders = [], []
    for p in tasks_paths:
        tr, te = load_task(p)
        train_loaders.append(tr)
        test_loaders.append(te)

    # ----- MODEL -----
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

    model = Model(Config()).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    der = DERContinualSMamba(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        replay_buffer_size=500,
        alpha=1.0,
        beta=1,
        replay_mode="logits"  # labels | logits | both
    )

    for t_idx, train_loader in enumerate(train_loaders):
        print(f"\n=== TRAIN TASK {t_idx+1} ===")
        der.fit_one_task(
            train_loader,
            label_len=LABEL_LEN,
            pred_len=PRED_LEN,
            task_idx=t_idx,
            epochs=EPOCHS
        )


if __name__ == "__main__":
    main()
