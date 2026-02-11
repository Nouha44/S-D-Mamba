import torch
import numpy as np
import pandas as pd
import random
from torch import nn
from torch.utils.data import DataLoader

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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================= DATASET =================
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
    ds = WeatherDataset(series.reshape(-1, 1), SEQ_LEN, PRED_LEN)

    split = int(0.8 * len(ds))
    train_loader = DataLoader(
        torch.utils.data.Subset(ds, range(split)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(ds, range(split, len(ds))),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return train_loader, test_loader

# ================= EVALUATION =================
def evaluate_rmse(model, dataloader):
    model.eval()
    mse_sum, n = 0.0, 0

    with torch.no_grad():
        for x, x_mark, y, y_mark in dataloader:
            x = x.to(DEVICE)
            x_mark = x_mark.to(DEVICE)
            y = y.to(DEVICE)
            y_mark = y_mark.to(DEVICE)

            dec_inp = torch.cat(
                [y[:, :LABEL_LEN, :], torch.zeros_like(y[:, LABEL_LEN:, :])],
                dim=1
            )

            preds = model(x, x_mark, dec_inp, y_mark)
            preds = preds[:, -PRED_LEN:, :]

            mse_sum += nn.functional.mse_loss(preds, y, reduction="sum").item()
            n += y.numel()

    return np.sqrt(mse_sum / n)

# ================= MAIN =================
def main():
    set_seed(42)

    task_paths = [
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
        freq = "m"
        output_attention = False
        use_norm = True
        class_strategy = "projection"

    for task_id, path in enumerate(task_paths):
        print(f"\n=== TRAINING DATASET {task_id + 1} (SINGLE TASK, NO CL) ===")

        train_loader, test_loader = load_task(path)

        # sample for model initialization
        x_sample, _, y_sample, _ = next(iter(train_loader))

        # -------- NEW MODEL EACH TIME --------
        trainer = DERContinualSMamba(
            config=Config(),
            replay_buffer_size=0,   # ❌ no replay
            alpha=0.0,              # ❌ no DER loss
            beta=0.0,
            replay_mode=None,
            device=DEVICE
        )

        trainer.create_network(x_sample, y_sample)

        optimizer = torch.optim.AdamW(trainer.network.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # -------- TRAIN AS SINGLE TASK --------
        trainer.fit_one_task(
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            label_len=LABEL_LEN,
            pred_len=PRED_LEN,
            task_idx=0,
            epochs=EPOCHS
        )

        # -------- EVALUATION --------
        rmse = evaluate_rmse(trainer.network, test_loader)
        print(f"RMSE Dataset {task_id + 1}: {rmse:.6f}")

        results.append(rmse)

    # -------- SAVE RESULTS --------
    df = pd.DataFrame(
        [results],
        columns=[f"Dataset{i + 1}" for i in range(len(results))]
    )
    df.to_csv("single_task_smamba_results.csv", index=False)

    print("\n=== SINGLE TASK RESULTS ===")
    print(df)

# ================= RUN =================
if __name__ == "__main__":
    main()
