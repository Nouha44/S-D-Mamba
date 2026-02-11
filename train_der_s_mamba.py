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

set_seed(42)

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
    return (
        DataLoader(torch.utils.data.Subset(ds, range(split)), batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        DataLoader(torch.utils.data.Subset(ds, range(split, len(ds))), batch_size=BATCH_SIZE, num_workers=0)
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

    # ---- TASK PATHS ----
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

    num_tasks = len(train_loaders)

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

    # ----- INITIAL MODEL DER -----
    der = DERContinualSMamba(
        config=Config(),
        replay_buffer_size=200,
        alpha=1.0,
        beta=0.0,
        replay_mode="logits",
        device=DEVICE
    )

    results_matrix = np.full((num_tasks, num_tasks), np.nan)
    if der.network is None:
        x_sample, x_mark_sample, y_sample, y_mark_sample = next(iter(train_loaders[0]))
        der.create_network(x_sample, y_sample)

    optimizer = torch.optim.AdamW(der.network.parameters(), lr=1e-3)
    # ================= TRAIN TASKS =================
    for t_idx, train_loader in enumerate(train_loaders):
        print(f"\n=== TRAIN TASK {t_idx+1} ===")
        der.fit_one_task(
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            label_len=LABEL_LEN,
            pred_len=PRED_LEN,
            task_idx=t_idx,
            epochs=EPOCHS
        )

        # --------- EVALUATION ---------
        for eval_idx in range(t_idx + 1):
            rmse = evaluate_rmse(der.network, test_loaders[eval_idx], LABEL_LEN, PRED_LEN, DEVICE)
            results_matrix[t_idx, eval_idx] = rmse
            print(f"RMSE Task{eval_idx+1} after Task{t_idx+1}: {rmse:.6f}")

    # ================= RESULTS =================
    df = pd.DataFrame(
        results_matrix,
        columns=[f"Task{i+1}" for i in range(num_tasks)],
        index=[f"after T{i+1}" for i in range(num_tasks)]
    )
    print("\n=== Global RMSE Results ===")
    print(df)
    df.to_csv("der_results.csv")

    # ================= BWT =================
    final_rmse = np.nanmean(results_matrix[-1])
    print(f"\n📊 Final Average RMSE (after all tasks): {final_rmse:.6f}")

    bwt_values = [results_matrix[-1, i] - results_matrix[i, i] for i in range(num_tasks-1)]
    bwt = np.mean(bwt_values)
    print(f"📉 Backward Transfer (BWT): {bwt:.6f}")

    # ================= IMPACT DU BUFFER =================
    buffer_sizes = [0, 150, 200, 300, 500, 1000, 1500, 2000, 2500]
    rmse_totals = []
    rmse_per_task = np.zeros((num_tasks, len(buffer_sizes)))
    bwt_totals = []
    bwt_per_task = np.zeros((num_tasks-1, len(buffer_sizes)))

    for b_idx, buf_size in enumerate(buffer_sizes):
        set_seed(42)
        torch.cuda.empty_cache()
        print(f"\n--- Training with buffer size = {buf_size} ---")

        # Nouveau DER avec buffer modifié
        der_buf = DERContinualSMamba(
            config=Config(),
            replay_buffer_size=buf_size,
            alpha=1.0,
            beta=0.0,
            replay_mode="logits",
            device=DEVICE
        )
        if der_buf.network is None:
            x_sample, x_mark_sample, y_sample, y_mark_sample = next(iter(train_loaders[0]))
            der_buf.create_network(x_sample, y_sample)

        optimizer = torch.optim.AdamW(der_buf.network.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        rmse_after_each_task = []
        for t_idx, task in enumerate(train_loaders):

            der_buf.fit_one_task(
                task,
                optimizer=optimizer,
                criterion=criterion,
                label_len=LABEL_LEN,
                pred_len=PRED_LEN,
                task_idx=t_idx,
                epochs=EPOCHS
            )
            rmse_task = evaluate_rmse(der_buf.network, test_loaders[t_idx], LABEL_LEN, PRED_LEN, DEVICE)
            rmse_after_each_task.append(rmse_task)

        # RMSE finale toutes tâches
        rmse_final_all_tasks = []
        for t_idx, task in enumerate(test_loaders):
            rmse_task = evaluate_rmse(der_buf.network, task, LABEL_LEN, PRED_LEN, DEVICE)
            rmse_per_task[t_idx, b_idx] = rmse_task
            rmse_final_all_tasks.append(rmse_task)
        rmse_totals.append(np.mean(rmse_final_all_tasks))

        # BWT
        bwt_values_buffer = []
        for task_id in range(num_tasks-1):
            bwt_task = rmse_final_all_tasks[task_id] - rmse_after_each_task[task_id]
            bwt_per_task[task_id, b_idx] = bwt_task
            bwt_values_buffer.append(bwt_task)
        bwt_totals.append(np.mean(bwt_values_buffer))

        print(f"Buffer {buf_size} -> Final Avg RMSE: {rmse_totals[-1]:.6f}, BWT: {bwt_totals[-1]:.6f}")

    # ================= PLOTS =================
    plt.figure(figsize=(8,4))
    plt.plot(buffer_sizes, rmse_totals, 'o-', color='blue', linewidth=2)
    plt.xlabel("Replay Buffer Size")
    plt.ylabel("Final Average RMSE")
    plt.title("Impact of Replay Buffer Size on Overall RMSE")
    plt.grid(True)
    plt.savefig('RMSE_total_der.png')
    plt.show()

    plt.figure(figsize=(8,4))
    colors = ['blue', 'green', 'red', 'orange']
    for t_idx in range(num_tasks):
        plt.plot(buffer_sizes, rmse_per_task[t_idx], 'o-', color=colors[t_idx], label=f'Task {t_idx+1}')
    plt.xlabel("Replay Buffer Size")
    plt.ylabel("Final RMSE per Task")
    plt.title("Impact of Buffer Size on RMSE per Task (After All Tasks)")
    plt.legend()
    plt.grid(True)
    plt.savefig('RMSE_per_task_der.png')
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(buffer_sizes, bwt_totals, 'o-', color='black', linewidth=2, label='BWT total')
    colors = ['blue', 'green', 'red']
    for t_idx in range(num_tasks-1):
        plt.plot(buffer_sizes, bwt_per_task[t_idx], 'o--', color=colors[t_idx], label=f'BWT Task {t_idx+1}')
    plt.xlabel("Replay Buffer Size")
    plt.ylabel("Backward Transfer (BWT)")
    plt.title("Evolution of Backward Transfer with Buffer Size")
    plt.legend()
    plt.grid(True)
    plt.savefig('backward_transfer_der.png')
    plt.show()


if __name__ == "__main__":
    main()
