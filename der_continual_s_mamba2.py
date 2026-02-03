import torch
import numpy as np
import random
from torch import nn


class DERContinualSMamba2:
    """
    Dark Experience Replay (DER / DER++)
    pour un modèle S-Mamba déjà défini
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        replay_buffer_size=500,
        alpha=1.0,
        beta=0.5,
        replay_mode="labels"
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.replay_buffer_size = replay_buffer_size
        self.alpha = alpha
        self.beta = beta
        self.replay_mode = replay_mode

        self.replay_buffer = []

    # -----------------------
    # Reservoir Sampling
    # -----------------------
    def reservoir_sampling(self, buffer, new_item):
        if self.replay_buffer_size == 0:
            return
        if len(buffer) < self.replay_buffer_size:
            buffer.append(new_item)
        else:
            idx = random.randint(0, len(buffer) - 1)
            if idx < self.replay_buffer_size:
                buffer[idx] = new_item

    # -----------------------
    # Entraînement 1 tâche
    # -----------------------
    def fit_one_task(self, train_loader, label_len, pred_len, task_idx=0, epochs=5):

        self.model.train()

        for epoch in range(epochs):
            task_losses, replay_losses = [], []

            for x, x_mark, y, y_mark in train_loader:
                x, x_mark = x.to(self.device), x_mark.to(self.device)
                y, y_mark = y.to(self.device), y_mark.to(self.device)

                dec_inp = torch.cat(
                    [y[:, :label_len, :], torch.zeros_like(y[:, label_len:, :])],
                    dim=1
                )

                # -------- forward tâche courante --------
                preds = self.model(x, x_mark, dec_inp, y_mark)
                preds = preds[:, -pred_len:, :]
                loss = self.criterion(preds, y)
                task_losses.append(loss.item())

                preds_detached = preds.detach().cpu()

                # -------- replay (UNIQUEMENT tâches futures) --------
                if task_idx > 0 and len(self.replay_buffer) > 0:

                    replay_idxs = np.random.choice(
                        len(self.replay_buffer),
                        min(len(self.replay_buffer), x.shape[0]),
                        replace=False
                    )

                    replay_samples = [self.replay_buffer[i] for i in replay_idxs]

                    x_mem = torch.stack([s[0] for s in replay_samples]).to(self.device)
                    x_mark_mem = torch.stack([s[1] for s in replay_samples]).to(self.device)
                    y_mem = torch.stack([s[2] for s in replay_samples]).to(self.device)
                    y_mark_mem = torch.stack([s[3] for s in replay_samples]).to(self.device)

                    dec_inp_mem = torch.cat(
                        [y_mem[:, :label_len, :],
                         torch.zeros_like(y_mem[:, label_len:, :])],
                        dim=1
                    )

                    y_mem_now = self.model(x_mem, x_mark_mem, dec_inp_mem, y_mark_mem)
                    y_mem_now = y_mem_now[:, -pred_len:, :]

                    replay_loss = nn.functional.mse_loss(y_mem_now, y_mem)
                    loss = loss + self.alpha * replay_loss
                    replay_losses.append(replay_loss.item())

                # -------- backward --------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # -------- stocker en mémoire (TOUTES les tâches) --------
                for xb, xb_mark, yb, yb_mark in zip(
                    x.cpu(), x_mark.cpu(), y.cpu(), y_mark.cpu()
                ):
                    item = (xb, xb_mark, yb, yb_mark)
                    self.reservoir_sampling(self.replay_buffer, item)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"TaskLoss={np.mean(task_losses):.4f} | "
                f"ReplayLoss={np.mean(replay_losses) if replay_losses else 0:.4f}"
            )

        print(f"Replay buffer size: {len(self.replay_buffer)}")
