import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import random

from der_continual_s_mamba2 import DERContinualSMamba  # ta classe DER pour S-Mamba

# ----------------------- UTILITAIRES -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_csv_as_series(path):
    df = pd.read_csv(path)
    return df['values'].values.astype(np.float32)

def create_windows(series, seq_len=256, pred_horizon=128):
    """
    Crée x_enc, x_mark, dec_inp, y_mark, y pour S-Mamba
    Shapes attendues par S-Mamba :
        x, x_mark: (num_samples, seq_len, input_dim)
        y, y_mark, dec_inp: (num_samples, pred_horizon, output_dim)
    """
    x_list, y_list = [], []
    total_len = seq_len + pred_horizon
    for i in range(len(series) - total_len):
        x_list.append(series[i:i+seq_len])
        y_list.append(series[i+seq_len:i+total_len])

    # Convertir en numpy float32
    x = np.array(x_list, dtype=np.float32)       # (num_samples, seq_len)
    y = np.array(y_list, dtype=np.float32)       # (num_samples, pred_horizon)

    # Ajouter la dimension "feature" à la fin
    x = x[..., np.newaxis]   # (num_samples, seq_len, 1)
    y = y[..., np.newaxis]   # (num_samples, pred_horizon, 1)

    x_mark = x.copy()        # timestamp / covariates
    y_mark = y.copy()

    # Decoder input : première moitié vraie, deuxième moitié zéro
    dec_inp = np.concatenate([y[:, :pred_horizon//2, :],
                              np.zeros_like(y[:, pred_horizon//2:, :])], axis=1)

    return x, x_mark, dec_inp, y_mark, y


def evaluate_rmse(model, x, x_mark, dec_inp, y_mark, y, device):
    model.model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32).to(device)
        x_mark_t = torch.tensor(x_mark, dtype=torch.float32).to(device)
        dec_t = torch.tensor(dec_inp, dtype=torch.float32).to(device)
        y_mark_t = torch.tensor(y_mark, dtype=torch.float32).to(device)
        y_t = torch.tensor(y, dtype=torch.float32).to(device)
        preds = model.model(x_t, x_mark_t, dec_t, y_mark_t)
        mse = nn.functional.mse_loss(preds, y_t).item()
        return np.sqrt(mse), preds.cpu().numpy()

def plot_forecast(history, y_true, y_pred, seq_len, pred_horizon, title):
    t_hist = np.arange(seq_len)
    t_pred = np.arange(seq_len, seq_len + pred_horizon)
    plt.figure(figsize=(8,4))
    plt.plot(t_hist, history.squeeze(), label='History')
    plt.plot(t_pred, y_true.squeeze(), 'o-', label='True')
    plt.plot(t_pred, y_pred.squeeze(), 'x--', label='Predicted')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------- MAIN -----------------------
def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_paths = [
        "/home/nkaraoul/timesfm_backup/mult_sin_d1_full.csv",
        "/home/nkaraoul/timesfm_backup/mult_sin_d2_full.csv",
        "/home/nkaraoul/timesfm_backup/mult_sin_d3_full.csv",
        "/home/nkaraoul/timesfm_backup/mult_sin_d4_full.csv",
    ]

    seq_len, pred_horizon = 256, 128
    tasks = []

    for path in dataset_paths:
        series = load_csv_as_series(path)
        x, x_mark, dec_inp, y_mark, y = create_windows(series, seq_len, pred_horizon)
        split = int(0.8 * len(x))
        tasks.append({
            "x_train": x[:split], "x_mark_train": x_mark[:split],
            "dec_train": dec_inp[:split], "y_mark_train": y_mark[:split],
            "y_train": y[:split],
            "x_test": x[split:], "x_mark_test": x_mark[split:],
            "dec_test": dec_inp[split:], "y_mark_test": y_mark[split:],
            "y_test": y[split:]
        })

    num_tasks = len(tasks)

    # ---------------- Initial Training ----------------
    print("\n=== Training on all tasks (initial model) ===")
    model = DERContinualSMamba(
        model=None,  # S-Mamba model added après
        learning_rate=1e-3,
        batch_size=64,
        num_epochs=5,
        replay_buffer_size=300,
        alpha=1.0,
        beta=None,
        replay_mode="logits",
        device=device
    )

    # --------------- Instantiate S-Mamba ---------------
    from model.S_Mamba import Model as SMambaModel
    from types import SimpleNamespace
    configs = SimpleNamespace(
        seq_len=seq_len,
        pred_len=pred_horizon,
        output_attention=False,
        use_norm=True,
        d_model=32,
        d_state=16,
        embed='timeF',
        freq='h',
        dropout=0.1,
        e_layers=2,
        d_ff=64,
        activation='gelu',
        class_strategy=None
    )
    model.model = SMambaModel(configs).to(device)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=model.learning_rate)
    criterion = nn.MSELoss().to(device)

    probe_samples = [(task["x_test"][0], task["y_test"][0]) for task in tasks]
    results_matrix = np.full((num_tasks, num_tasks), np.nan)

    for t_idx, task in enumerate(tasks):
        print(f"\n--- Training Task {t_idx+1} ---")
        model.fit_one_task(task["x_train"], task["x_mark_train"], task["dec_train"],
                           task["y_mark_train"], optimizer, criterion, task_idx=t_idx)

        # Replay buffer composition
        print(f"\n📦 Replay buffer size: {model.buffer_filled}")
        if model.task_id_buffer is not None:
            for task_id in range(num_tasks):
                n_examples = (model.task_id_buffer[:model.buffer_filled] == task_id).sum().item()
                print(f"  Task {task_id+1}: {n_examples} exemples dans le buffer")

        # Evaluation
        for eval_idx in range(t_idx+1):
            rmse, preds = evaluate_rmse(model,
                                        tasks[eval_idx]["x_test"],
                                        tasks[eval_idx]["x_mark_test"],
                                        tasks[eval_idx]["dec_test"],
                                        tasks[eval_idx]["y_mark_test"],
                                        tasks[eval_idx]["y_test"],
                                        device)
            results_matrix[t_idx, eval_idx] = rmse
            print(f"RMSE Task{eval_idx+1} after Task{t_idx+1}: {rmse:.4f}")

            # Probe plot
            x_probe, y_probe = probe_samples[eval_idx]
            x_t = torch.tensor(x_probe, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                y_pred = model.model(x_t, x_t, x_t, x_t).squeeze(0).cpu().numpy()
            plot_forecast(x_probe, y_probe, y_pred, seq_len, pred_horizon,
                          title=f"Probe Task{eval_idx+1} after Task{t_idx+1}")

    # ---------------- Metrics and Plots ----------------
    df = pd.DataFrame(results_matrix,
                      columns=[f"Task{i+1}" for i in range(num_tasks)],
                      index=[f"after T{i+1}" for i in range(num_tasks)])
    print("\n=== Global RMSE Results ===")
    print(df)
    df.to_csv("der_s_mamba_results.csv", index=True)

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

    # Impact buffer size
    buffer_sizes = [0, 150 ,300, 500, 1000, 1500, 2000, 2500]
    rmse_totals = []
    rmse_per_task = np.zeros((num_tasks, len(buffer_sizes)))
    bwt_totals = []
    bwt_per_task = np.zeros((num_tasks-1, len(buffer_sizes)))

    for b_idx, buf_size in enumerate(buffer_sizes):
        set_seed(42)
        print(f"\n--- Training with buffer size = {buf_size} ---")
        model = DERContinualSMamba(
            model=SMambaModel(configs).to(device),
            learning_rate=1e-3,
            batch_size=64,
            num_epochs=5,
            replay_buffer_size=buf_size,
            alpha=2,
            beta=None,
            replay_mode="logits",
            device=device
        )
        optimizer = torch.optim.Adam(model.model.parameters(), lr=model.learning_rate)
        criterion = nn.MSELoss().to(device)

        rmse_after_each_task = []

        for t_idx, task in enumerate(tasks):
            model.fit_one_task(task["x_train"], task["x_mark_train"], task["dec_train"],
                               task["y_mark_train"], optimizer, criterion, task_idx=t_idx)
            rmse_task, _ = evaluate_rmse(model, task["x_test"], task["x_mark_test"],
                                         task["dec_test"], task["y_mark_test"], task["y_test"], device)
            rmse_after_each_task.append(rmse_task)

        rmse_final_all_tasks = []
        for t_idx, task in enumerate(tasks):
            rmse_task, _ = evaluate_rmse(model, task["x_test"], task["x_mark_test"],
                                         task["dec_test"], task["y_mark_test"], task["y_test"], device)
            rmse_per_task[t_idx, b_idx] = rmse_task
            rmse_final_all_tasks.append(rmse_task)
        rmse_totals.append(np.mean(rmse_final_all_tasks))

        bwt_values_buffer = []
        for task_id in range(num_tasks-1):
            perf_after_learned = rmse_after_each_task[task_id]
            perf_after_final = rmse_final_all_tasks[task_id]
            bwt_task = perf_after_final - perf_after_learned
            bwt_per_task[task_id, b_idx] = bwt_task
            bwt_values_buffer.append(bwt_task)
        bwt_totals.append(np.mean(bwt_values_buffer))
        print(f"Buffer {buf_size} -> Final Avg RMSE: {rmse_totals[-1]:.4f}, BWT: {bwt_totals[-1]:.4f}")

    # Plots (comme dans ton script SAMFormer)
    plt.figure(figsize=(8,4))
    plt.plot(buffer_sizes, rmse_totals, 'o-', color='blue', linewidth=2)
    plt.xlabel("Replay Buffer Size")
    plt.ylabel("Final Average RMSE")
    plt.title("Impact of Replay Buffer Size on Overall RMSE")
    plt.grid(True)
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
    plt.show()


if __name__ == "__main__":
    main()
