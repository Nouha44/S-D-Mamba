import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from torch import nn
from torch.utils.data import DataLoader

from model.S_Mamba import Model

# ----------------------- UTILITAIRES -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEQ_LEN = 256
PRED_LEN = 128
LABEL_LEN = SEQ_LEN // 2
BATCH_SIZE = 64
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    ds = WeatherDataset(series.reshape(-1,1), SEQ_LEN, PRED_LEN)
    split = int(0.8 * len(ds))
    return (
        DataLoader(torch.utils.data.Subset(ds, range(split)), batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(torch.utils.data.Subset(ds, range(split, len(ds))), batch_size=BATCH_SIZE)
    )

def evaluate_rmse(model, test_loader):
    model.eval()
    preds_list, true_list, history_list = [], [], []
    with torch.no_grad():
        for x, x_mark, y, y_mark in test_loader:
            x, x_mark = x.to(DEVICE), x_mark.to(DEVICE)
            y, y_mark = y.to(DEVICE), y_mark.to(DEVICE)
            dec_inp = torch.cat([y[:, :LABEL_LEN, :], torch.zeros_like(y[:, LABEL_LEN:, :])], dim=1)
            y_pred = model(x, x_mark, dec_inp, y_mark)[:, -PRED_LEN:, :]
            preds_list.append(y_pred.cpu())
            true_list.append(y.cpu())
            history_list.append(x.cpu())
    preds_all = torch.cat(preds_list, dim=0).numpy()
    true_all = torch.cat(true_list, dim=0).numpy()
    history_all = torch.cat(history_list, dim=0).numpy()
    rmse = np.sqrt(np.mean((preds_all - true_all)**2))
    return rmse, preds_all, true_all, history_all

# ----------------------- MAIN -----------------------
def main():
    set_seed(42)

    # ----- TASKS PATHS -----
    tasks_paths = [
        "/home/nkaraoul/timesfm_backup/mult_sin_d1_full.csv",
        "/home/nkaraoul/timesfm_backup/mult_sin_d2_full.csv",
        "/home/nkaraoul/timesfm_backup/mult_sin_d3_full.csv",
        "/home/nkaraoul/timesfm_backup/mult_sin_d4_full.csv",
    ]

    train_loaders, test_loaders = [], []
    for path in tasks_paths:
        tr, te = load_task(path)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_tasks = len(tasks_paths)
    results_matrix = np.full((num_tasks, num_tasks), np.nan)

    # ----- FINETUNING SEQUENTIEL -----
    for t_idx, train_loader in enumerate(train_loaders):
        print(f"\n=== Fine-tuning on Task {t_idx+1} ===")
        model.train()
        for epoch in range(EPOCHS):
            for x, x_mark, y, y_mark in train_loader:
                x, x_mark = x.to(DEVICE), x_mark.to(DEVICE)
                y, y_mark = y.to(DEVICE), y_mark.to(DEVICE)
                optimizer.zero_grad()
                dec_inp = torch.cat([y[:, :LABEL_LEN, :], torch.zeros_like(y[:, LABEL_LEN:, :])], dim=1)
                y_pred = model(x, x_mark, dec_inp, y_mark)[:, -PRED_LEN:, :]
                loss = criterion(y_pred, y[:, -PRED_LEN:, :])
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{EPOCHS} done")

        # ----- EVALUER SUR TOUTES LES TÂCHES VUES -----
        for eval_idx in range(t_idx+1):
            rmse, _, _, _ = evaluate_rmse(model, test_loaders[eval_idx])
            results_matrix[t_idx, eval_idx] = rmse
            print(f"RMSE Task{eval_idx+1} after fine-tuning on Task{t_idx+1}: {rmse:.4f}")

    # ----- SAVE RESULTS -----
    df = pd.DataFrame(results_matrix,
                      columns=[f"Task{i+1}" for i in range(num_tasks)],
                      index=[f"after T{i+1}" for i in range(num_tasks)])
    df.to_csv("finetune_sequential_all_tasks.csv")
    print("\n=== RMSE Results ===")
    print(df)

if __name__ == "__main__":
    main()
