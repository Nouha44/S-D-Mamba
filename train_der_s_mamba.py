import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

from model.S_Mamba import Model
from der_continual_s_mamba2 import DERContinualSMamba  # nouvelle version stricte
import random

# ---------------- CONFIG ----------------
SEQ_LEN = 256
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
        # x_mark et y_mark remplis à 0 pour S-Mamba
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- EVALUATION ----------------
def evaluate_rmse(model, dataloader, label_len, pred_len, device):
    model.eval()
    mse_sum, n = 0.0, 0

    with torch.no_grad():
        for x, x_mark, y, y_mark in dataloader:
            x = x.to(device)
            x_mark = x_mark.to(device)
            y = y.to(device)
            y_mark = y_mark.to(device)

            dec_inp = torch.cat(
                [y[:, :label_len, :],
                 torch.zeros_like(y[:, label_len:, :])],
                dim=1
            )

            preds = model(x, x_mark, dec_inp, y_mark)
            preds = preds[:, -pred_len:, :]

            mse_sum += nn.functional.mse_loss(preds, y, reduction="sum").item()
            n += y.numel()

    return np.sqrt(mse_sum / n)


# ---------------- MAIN ----------------
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

    num_tasks = len(train_loaders)

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
        output_attention = False
        freq= 'm'
        use_norm = True
        class_strategy = "projection"

    model = Model(Config()).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # ----- DER Continual Learning -----
    der = DERContinualSMamba(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        replay_buffer_size=200,
        alpha=1.0,
        beta=0,  # DER++ mixing parameter
        replay_mode="logits"  # "labels", "logits" ou "both"
    )

    # ================= METRICS =================
    results_matrix = np.full((num_tasks, num_tasks), np.nan)

    # ================= TRAIN LOOP =================
    for t_idx, train_loader in enumerate(train_loaders):
        print(f"\n=== TRAIN TASK {t_idx+1} ===")

        der.fit_one_task(
            train_loader,
            label_len=LABEL_LEN,
            pred_len=PRED_LEN,
            task_idx=t_idx,
            epochs=EPOCHS
        )

        # --------- EVALUATION ---------
        for eval_idx in range(t_idx + 1):
            rmse = evaluate_rmse(
                model,
                test_loaders[eval_idx],
                LABEL_LEN,
                PRED_LEN,
                DEVICE
            )
            results_matrix[t_idx, eval_idx] = rmse
            print(f"RMSE Task{eval_idx+1} after Task{t_idx+1}: {rmse:.4f}")

    # ================= RESULTS =================
    df = pd.DataFrame(
        results_matrix,
        columns=[f"Task{i+1}" for i in range(num_tasks)],
        index=[f"after T{i+1}" for i in range(num_tasks)]
    )

    print("\n=== Global RMSE Results ===")
    print(df)
    df.to_csv("der_results.csv")

    # ================= CONTINUAL METRICS =================
    # Final Average RMSE
    final_rmse = np.nanmean(results_matrix[-1])
    print(f"\n📊 Final Average RMSE (after T{num_tasks}): {final_rmse:.4f}")

    # Backward Transfer (BWT)
    # ----- METRICS CONTINUAL -----
    final_rmse = np.nanmean(results_matrix[-1])
    print(f"\n📊 Final Average RMSE (after all tasks): {final_rmse:.4f}")

    bwt_values = []
    for task_id in range(num_tasks-1):
        perf_after_learned = results_matrix[task_id, task_id]
        perf_after_final = results_matrix[-1, task_id]
        if not np.isnan(perf_after_learned) and not np.isnan(perf_after_final):
            bwt_values.append(perf_after_final - perf_after_learned)
    bwt = np.mean(bwt_values) if len(bwt_values) > 0 else np.nan
    print(f"📉 Backward Transfer (BWT): {bwt:.4f}")

    # ----- IMPACT DE LA TAILLE DU BUFFER -----
    buffer_sizes = [0, 150 ,300, 500, 1000, 1500, 2000, 2500]
    rmse_totals = []
    rmse_per_task = np.zeros((num_tasks, len(buffer_sizes)))
    bwt_totals = []
    bwt_per_task = np.zeros((num_tasks-1, len(buffer_sizes)))

    for b_idx, buf_size in enumerate(buffer_sizes):
        set_seed(42)
        print(f"\n--- Training with buffer size = {buf_size} ---")
        model = DERContinualSAMFormer(
            learning_rate=1e-3,
            batch_size=64,
            num_epochs=5,
            replay_buffer_size=buf_size,
            alpha=1,
            beta=None,
            replay_mode="logits",
            device=device
        )
        model.create_network(tasks[0]["x_train"], tasks[0]["y_train"])
        optimizer = torch.optim.Adam(model.network.parameters(), lr=model.learning_rate)
        criterion = nn.MSELoss().to(device)

        # RMSE juste après chaque tâche
        rmse_after_each_task = []

        for t_idx, task in enumerate(tasks):
            model.fit_one_task(task["x_train"], task["y_train"], optimizer, criterion, task_idx=t_idx)
            rmse_task, _ = evaluate_rmse(model, task["x_test"], task["y_test"], device)
            rmse_after_each_task.append(rmse_task)

        # RMSE finale pour chaque tâche après tout l’apprentissage
        rmse_final_all_tasks = []
        for t_idx, task in enumerate(tasks):
            rmse_task, _ = evaluate_rmse(model, task["x_test"], task["y_test"], device)
            rmse_per_task[t_idx, b_idx] = rmse_task
            rmse_final_all_tasks.append(rmse_task)

        rmse_totals.append(np.mean(rmse_final_all_tasks))

        # Calcul BWT réel pour chaque tâche
        bwt_values_buffer = []
        for task_id in range(num_tasks-1):
            perf_after_learned = rmse_after_each_task[task_id]
            perf_after_final = rmse_final_all_tasks[task_id]
            bwt_task = perf_after_final - perf_after_learned
            bwt_per_task[task_id, b_idx] = bwt_task
            bwt_values_buffer.append(bwt_task)

        bwt_totals.append(np.mean(bwt_values_buffer))
        print(f"Buffer {buf_size} -> Final Avg RMSE: {rmse_totals[-1]:.4f}, BWT: {bwt_totals[-1]:.4f}")

    # ----- PLOTS -----
    # Plot 1: RMSE totale vs buffer
    plt.figure(figsize=(8,4))
    plt.plot(buffer_sizes, rmse_totals, 'o-', color='blue', linewidth=2)
    plt.xlabel("Replay Buffer Size")
    plt.ylabel("Final Average RMSE")
    plt.title("Impact of Replay Buffer Size on Overall RMSE")
    plt.grid(True)
    plt.show()

    # Plot 2: RMSE par tâche vs buffer
    plt.figure(figsize=(8,4))
    colors = ['blue', 'green', 'red', 'orange']
    for t_idx in range(num_tasks):
        plt.plot(buffer_sizes, rmse_per_task[t_idx], 'o-', color=colors[t_idx], label=f'Task {t_idx+1}')
    plt.xlabel("Replay Buffer Size")
    plt.ylabel("Final RMSE per Task")
    plt.title("Impact of Buffer Size on RMSE per Task (After All Tasks)")
    plt.legend()
    plt.grid(True)
    plt.save('RMSE_der.png')
    plt.show()

    # Plot 3: BWT vs buffer
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
    plt.save('backward_transfer_der.png')
    plt.show()

if __name__ == "__main__":
    main()
