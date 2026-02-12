# train_cumulative_smamba.py
import torch
import numpy as np
import pandas as pd
import random
from torch import nn
from torch.utils.data import DataLoader, Subset
from model.S_Mamba import Model
from der_continual_s_mamba2 import DERContinualSMamba

# ================= CONFIG =================
SEQ_LEN = 256
PRED_LEN = 128
LABEL_LEN = SEQ_LEN // 2
BATCH_SIZE = 64
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= SEED =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    random.seed(42)

# ================= DATASET =================
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
    train_gen = torch.Generator(); train_gen.manual_seed(42)
    test_gen = torch.Generator(); test_gen.manual_seed(42)
    return (
        DataLoader(Subset(ds, range(split)), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=train_gen),
        DataLoader(Subset(ds, range(split, len(ds))), batch_size=BATCH_SIZE, num_workers=0, generator=test_gen)
    )

# ================= EVALUATION =================
def evaluate_rmse(model, dataloader, label_len, pred_len, device):
    model.eval()
    mse_sum, n = 0.0, 0
    with torch.no_grad():
        for x, x_mark, y, y_mark in dataloader:
            x, x_mark, y, y_mark = x.to(device), x_mark.to(device), y.to(device), y_mark.to(device)
            dec_inp = torch.cat([y[:, :label_len, :], torch.zeros_like(y[:, label_len:, :])], dim=1)
            preds = model(x, x_mark, dec_inp, y_mark)
            preds = preds[:, -pred_len:, :]
            mse_sum += nn.functional.mse_loss(preds, y, reduction="sum").item()
            n += y.numel()
    return np.sqrt(mse_sum / n)

# ================= MAIN =================
def main():
    set_seed(42)

    tasks_paths = [
        "/home/nkaraoul/timesfm_backup/mult_sin_d1_full.csv",
        "/home/nkaraoul/timesfm_backup/mult_sin_d2_full.csv",
        "/home/nkaraoul/timesfm_backup/mult_sin_d3_full.csv",
        "/home/nkaraoul/timesfm_backup/mult_sin_d4_full.csv",
    ]

    tasks = []
    for path in tasks_paths:
        train_loader, test_loader = load_task(path)
        tasks.append({"train_loader": train_loader, "test_loader": test_loader})

    num_tasks = len(tasks)

    # ----- MODEL CONFIG -----
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
        freq = 'm'
        output_attention = False
        use_norm = True
        class_strategy = "projection"

    # ----- CREATE S-MAMBA DER MODEL -----
    model = DERContinualSMamba(
        config=Config(),
        replay_buffer_size=5000,  # pour cumulatif, on peut stocker un buffer
        alpha=1.0,
        beta=0.0,
        replay_mode="logits",
        device=DEVICE
    )

    # Sample pour initialisation sur le premier task
    x_sample, x_mark_sample, y_sample, y_mark_sample = next(iter(tasks[0]["train_loader"]))
    model.create_network(x_sample, y_sample)

    optimizer = torch.optim.AdamW(model.network.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # ----- TRAIN TASK 1 -----
    print("\n=== Training Task 1 ===")
    model.fit_one_task(
        tasks[0]["train_loader"],
        optimizer,
        criterion,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        task_idx=0,
        epochs=EPOCHS
    )

    # ----- CUMULATIVE TRAINING -----
    results_matrix = np.full((num_tasks, num_tasks), np.nan)

    # RMSE Task1 après premier training
    rmse = evaluate_rmse(model, tasks[0]["test_loader"], LABEL_LEN, PRED_LEN, DEVICE)
    results_matrix[0, 0] = rmse
    print(f"RMSE Task1 after first training: {rmse:.6f}")

    for t_idx in range(1, num_tasks):
        print(f"\n=== Cumulative training on Tasks 1 → {t_idx+1} ===")
        # concat train loaders des tasks vus jusqu'ici
        cumulative_batches = []
        for i in range(t_idx + 1):
            for batch in tasks[i]["train_loader"]:
                cumulative_batches.append(batch)
        # créer un DataLoader unique cumulatif
        x_all = torch.cat([b[0] for b in cumulative_batches], dim=0)
        x_mark_all = torch.cat([b[1] for b in cumulative_batches], dim=0)
        y_all = torch.cat([b[2] for b in cumulative_batches], dim=0)
        y_mark_all = torch.cat([b[3] for b in cumulative_batches], dim=0)
        joint_dataset = torch.utils.data.TensorDataset(x_all, x_mark_all, y_all, y_mark_all)
        joint_loader = DataLoader(joint_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model.fit_one_task(
            joint_loader,
            optimizer,
            criterion,
            label_len=LABEL_LEN,
            pred_len=PRED_LEN,
            task_idx=t_idx,
            epochs=EPOCHS
        )

        # évaluation sur tous les tasks vus
        for eval_idx in range(t_idx + 1):
            rmse = evaluate_rmse(model, tasks[eval_idx]["test_loader"], LABEL_LEN, PRED_LEN, DEVICE)
            results_matrix[t_idx, eval_idx] = rmse
            print(f"RMSE Task{eval_idx+1} after cumulative T{t_idx+1}: {rmse:.6f}")

    # ----- SAVE RESULTS -----
    df_results = pd.DataFrame(
        results_matrix,
        columns=[f"Task{i+1}" for i in range(num_tasks)],
        index=[f"after T{i+1}" for i in range(num_tasks)]
    )
    df_results.to_csv("cumulative_smamba_results.csv", index=False)
    print("\n=== Cumulative RMSE Results ===")
    print(df_results)

if __name__ == "__main__":
    main()
