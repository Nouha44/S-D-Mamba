import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import random
from der_continual_s_mamba2 import DERContinualSMamba
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
    return df['lead_1'].values.astype(np.float32)

def create_windows(series, seq_len=60, pred_horizon=10):
    """
    Crée les entrées x, x_mark, dec_inp, y_mark, y pour S-Mamba
    """
    x_list, y_list = [], []
    total_len = seq_len + pred_horizon
    for i in range(len(series) - total_len):
        x_list.append(series[i:i+seq_len])
        y_list.append(series[i+seq_len:i+total_len])

    x = np.array(x_list).reshape(-1, 1, seq_len).astype(np.float32)       # encoder input
    y = np.array(y_list).reshape(-1, pred_horizon, 1).astype(np.float32)  # target

    # Pour S-Mamba, simplification : x_mark, dec_inp, y_mark = copies of x/y
    x_mark = x.copy()
    y_mark = y.copy()
    dec_inp = np.concatenate([y[:, :pred_horizon//2, :], np.zeros_like(y[:, pred_horizon//2:, :])], axis=1)

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
        "101_ecg.csv",
        "100_ecg.csv",
        "102_ecg.csv",
        "103_ecg.csv",
    ]

    seq_len, pred_horizon = 60, 10
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

    # ---------------- Training initial ----------------
    print("\n=== Training on all tasks (initial model) ===")
    model = DERContinualSMamba(
        model=None,  # initial model provided later
        learning_rate=1e-3,
        batch_size=64,
        num_epochs=5,
        replay_buffer_size=300,
        alpha=1.0,
        beta=None,
        replay_mode="logits",
        device=device
    )

    # Ici vous devez initialiser votre modèle S-Mamba
    from model.S_Mamba import Model as smm
    model.model = smm.SMambaArchitecture(input_size=1, seq_len=seq_len, pred_horizon=pred_horizon).to(device)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=model.learning_rate)
    criterion = nn.MSELoss().to(device)

    # Probe sample
    probe_samples = [(task["x_test"][0], task["y_test"][0]) for task in tasks]
    results_matrix = np.full((num_tasks, num_tasks), np.nan)

    for t_idx, task in enumerate(tasks):
        print(f"\n--- Training Task {t_idx+1} ---")
        model.fit_one_task(task["x_train"], task["x_mark_train"], task["dec_train"],
                           task["y_mark_train"], optimizer, criterion, task_idx=t_idx)

        # Afficher composition du buffer
        print(f"\n📦 Replay buffer size: {model.buffer_filled}")
        if model.task_id_buffer is not None:
            for task_id in range(num_tasks):
                n_examples = (model.task_id_buffer[:model.buffer_filled] == task_id).sum().item()
                print(f"  Task {task_id+1}: {n_examples} exemples dans le buffer")

        # Évaluation des tâches vues
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

            # Plot probe sample
            x_probe, y_probe = probe_samples[eval_idx]
            x_t = torch.tensor(x_probe, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                y_pred = model.model(x_t, x_t, x_t, x_t).squeeze(0).cpu().numpy()  # simplification
            plot_forecast(x_probe, y_probe, y_pred, seq_len, pred_horizon,
                          title=f"Probe Task{eval_idx+1} after Task{t_idx+1}")


if __name__ == "__main__":
    main()
