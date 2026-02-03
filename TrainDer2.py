import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from torch import nn
from torch.utils.data import DataLoader

from model.S_Mamba import Model
from der_continual_s_mamba2 import DERContinualSMamba2

# ----------------------- UTILITAIRES -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_forecast(history, y_true, y_pred, seq_len, pred_len, title, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    t_hist = np.arange(seq_len)
    t_pred = np.arange(seq_len, seq_len + pred_len)

    plt.figure(figsize=(8, 4))
    plt.plot(t_hist, history.squeeze(), label="History")
    plt.plot(t_pred, y_true.squeeze(), "o-", label="True")
    plt.plot(t_pred, y_pred.squeeze(), "x--", label="Predicted")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, title.replace(" ", "_") + ".png"))
    plt.close()


# ----------------------- DATASET -----------------------
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
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x, torch.zeros_like(x), y, torch.zeros_like(y)


def load_task(path):
    series = pd.read_csv(path)["values"].values.astype("float32")
    series = (series - series.mean()) / (series.std() + 1e-6)

    ds = WeatherDataset(series.reshape(-1, 1), SEQ_LEN, PRED_LEN)
    split = int(0.8 * len(ds))

    return (
        DataLoader(
            torch.utils.data.Subset(ds, range(split)),
            batch_size=BATCH_SIZE,
            shuffle=True,
        ),
        DataLoader(
            torch.utils.data.Subset(ds, range(split, len(ds))),
            batch_size=BATCH_SIZE,
        ),
    )


def evaluate_rmse(der, test_loader):
    der.model.eval()
    preds, trues, history = [], [], []

    with torch.no_grad():
        for x, x_mark, y, y_mark in test_loader:
            x, x_mark = x.to(DEVICE), x_mark.to(DEVICE)
            y, y_mark = y.to(DEVICE), y_mark.to(DEVICE)

            dec_inp = torch.cat(
                [y[:, :LABEL_LEN, :], torch.zeros_like(y[:, LABEL_LEN:, :])], dim=1
            )

            y_pred = der.model(x, x_mark, dec_inp, y_mark)[:, -PRED_LEN:, :]
            preds.append(y_pred.cpu())
            trues.append(y.cpu())
            history.append(x.cpu())

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    history = torch.cat(history).numpy()

    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    return rmse, preds, trues, history


# ----------------------- MAIN -----------------------
def main():
    set_seed(42)

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

    der = DERContinualSMamba2(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        replay_buffer_size=1500,
        alpha=1.0,
    )

    num_tasks = len(tasks_paths)
    results = np.full((num_tasks, num_tasks), np.nan)

    for t_idx, train_loader in enumerate(train_loaders):
        print(f"\n=== Training Task {t_idx+1} ===")
        der.fit_one_task(train_loader, LABEL_LEN, PRED_LEN, t_idx, EPOCHS)

        for eval_idx in range(t_idx + 1):
            rmse, preds, trues, history = evaluate_rmse(der, test_loaders[eval_idx])
            results[t_idx, eval_idx] = rmse
            print(f"RMSE Task{eval_idx+1} after Task{t_idx+1}: {rmse:.4f}")

            plot_forecast(
                history[0],
                trues[0],
                preds[0],
                SEQ_LEN,
                PRED_LEN,
                f"Probe_Task{eval_idx+1}_after_Task{t_idx+1}",
            )

    df = pd.DataFrame(
        results,
        columns=[f"Task{i+1}" for i in range(num_tasks)],
        index=[f"After_Task{i+1}" for i in range(num_tasks)],
    )
    df.to_csv("der_results.csv")
    print(df)


if __name__ == "__main__":
    main()
