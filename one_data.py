import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import random
from model.S_Mamba import Model  # ton S-Mamba
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
    train_gen = torch.Generator()
    train_gen.manual_seed(42)
    test_gen = torch.Generator()
    test_gen.manual_seed(42)
    return (
        DataLoader(torch.utils.data.Subset(ds, range(split)), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=train_gen),
        DataLoader(torch.utils.data.Subset(ds, range(split, len(ds))), batch_size=BATCH_SIZE, num_workers=0, generator=test_gen)
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

    results = []

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

    for task_id, path in enumerate(tasks_paths):
        print(f"\n=== TRAINING DATASET {task_id+1} (SINGLE TASK, NO CL) ===")

        train_loader, test_loader = load_task(path)

        # Sample pour l'initialisation du modèle
        x_sample, _, y_sample, _ = next(iter(train_loader))

        # -------- DER MODEL INITIALIZATION --------
        model = DERContinualSMamba(
            config=Config(),
            replay_buffer_size=0,  # ❌ no replay
            alpha=0.0,             # ❌ no DER loss
            beta=0.0,
            replay_mode=None,
            device=DEVICE
        )

        model.create_network(x_sample, y_sample)

        optimizer = torch.optim.AdamW(model.network.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # -------- TRAIN AS TASK 0 --------
        model.fit_one_task(
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            label_len=LABEL_LEN,
            pred_len=PRED_LEN,
            task_idx=0,  # ⚠️ toujours 0
            epochs=EPOCHS
        )

        # -------- EVALUATION --------
        rmse = evaluate_rmse(
            model.network,
            test_loader,
            LABEL_LEN,
            PRED_LEN,
            DEVICE
        )

        print(f"RMSE Dataset {task_id+1}: {rmse:.6f}")
        results.append(rmse)

    # -------- SAVE RESULTS --------
    df = pd.DataFrame(
        [results],
        columns=[f"Dataset{i+1}" for i in range(len(results))]
    )
    df.to_csv("single_task_smamba_results.csv", index=False)

    print("\n=== SINGLE TASK RESULTS ===")
    print(df)

# ================= RUN =================
if __name__ == "__main__":
    main()
