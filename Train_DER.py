import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from torch import nn
from torch.utils.data import DataLoader

from model.S_Mamba import Model
from der_continual_s_mamba import DERContinualSMamba

# ----------------------- UTILITAIRES -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_forecast(history, y_true, y_pred, seq_len, pred_len, title, save_dir="plots"):
    """
    history     : séquence passée (input)
    y_true      : vraie valeur future
    y_pred      : prédiction du modèle
    seq_len     : longueur de l'historique
    pred_len    : longueur de la prédiction
    title       : titre du plot
    save_dir    : dossier où enregistrer le plot
    """
    os.makedirs(save_dir, exist_ok=True)
    t_hist = np.arange(seq_len)
    t_pred = np.arange(seq_len, seq_len + pred_len)

    plt.figure(figsize=(8,4))
    plt.plot(t_hist, history.squeeze(), label='History')
    plt.plot(t_pred, y_true.squeeze(), 'o-', label='True')
    plt.plot(t_pred, y_pred.squeeze(), 'x--', label='Predicted')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    safe_title = title.replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(save_dir, f"{safe_title}.png"))
    plt.close()

# ----------------------- DATASET -----------------------
SEQ_LEN = 128
PRED_LEN = 128
LABEL_LEN = SEQ_LEN // 2
BATCH_SIZE = 64
EPOCHS = 5
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

def evaluate_rmse(model, test_loader, label_len=LABEL_LEN, pred_len=PRED_LEN):
    model.model.eval()
    preds_list, true_list, history_list = [], [], []
    with torch.no_grad():
        for x, x_mark, y, y_mark in test_loader:
            x, x_mark = x.to(DEVICE), x_mark.to(DEVICE)
            y, y_mark = y.to(DEVICE), y_mark.to(DEVICE)
            dec_inp = torch.cat([y[:, :label_len, :], torch.zeros_like(y[:, label_len:, :])], dim=1)
            y_pred = model.model(x, x_mark, dec_inp, y_mark)[:, -pred_len:, :]
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

    # ----- TASKS -----
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

    der = DERContinualSMamba(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        replay_buffer_size=0,
        alpha=1.0,
        replay_mode="logits"
    )

    # ----- RESULTS MATRIX -----
    num_tasks = len(tasks_paths)
    results_matrix = np.full((num_tasks, num_tasks), np.nan)

    # ----- PROBE SAMPLES -----
    probe_samples = []
    for test_loader in test_loaders:
        x_probe, _, y_probe, _ = next(iter(test_loader))
        probe_samples.append((x_probe[0].numpy(), y_probe[0].numpy()))

    # ----- TRAIN LOOP -----
    for t_idx, train_loader in enumerate(train_loaders):
        print(f"\n=== Training Task {t_idx+1} ===")
        der.fit_one_task(train_loader, label_len=LABEL_LEN, pred_len=PRED_LEN, task_idx=t_idx, epochs=EPOCHS)

        # Évaluer toutes les tâches vues jusqu'à présent
        for eval_idx in range(t_idx+1):
            rmse, preds, true_vals, history_vals = evaluate_rmse(der, test_loaders[eval_idx])
            results_matrix[t_idx, eval_idx] = rmse
            print(f"RMSE Task{eval_idx+1} after Task{t_idx+1}: {rmse:.4f}")

            # plot probe et enregistrer
            x_probe, y_probe = probe_samples[eval_idx]
            y_pred_probe = preds[0]
            plot_forecast(history_vals[0], y_probe, y_pred_probe, SEQ_LEN, PRED_LEN,
                          title=f"Probe Task{eval_idx+1} after Task{t_idx+1}", save_dir="plots")

    # ----- SAVE RESULTS -----
    df = pd.DataFrame(results_matrix,
                      columns=[f"Task{i+1}" for i in range(num_tasks)],
                      index=[f"after T{i+1}" for i in range(num_tasks)])
    df.to_csv("der_results.csv")
    print("\n=== Global RMSE Results ===")
    print(df)

    # ----- CALCULER METRIQUES CONTINUAL -----
    final_rmse = np.nanmean(results_matrix[-1])
    print(f"\n📊 Final Average RMSE (after T{num_tasks}): {final_rmse:.4f}")

    # BWT (Backward Transfer / Forgetting)
    bwt_values = []
    for task_id in range(num_tasks - 1):
        perf_after_learned = results_matrix[task_id, task_id]
        perf_after_final = results_matrix[-1, task_id]
        bwt_values.append(perf_after_final - perf_after_learned)
    bwt = np.mean(bwt_values)
    print(f"📉 Backward Transfer (BWT): {bwt:.4f}")

if __name__ == "__main__":
    main()
