import torch
import numpy as np
import random
from torch import nn


class DERContinualSMamba:
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

        self.replay_buffer = []  # mémoire DER

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

                # -------- replay --------
                if (
                    self.replay_buffer_size > 0
                    and len(self.replay_buffer) > 0
                    and task_idx > 0
                ):
                    replay_idxs = np.random.choice(
                        len(self.replay_buffer),
                        min(len(self.replay_buffer), x.shape[0]),
                        replace=False
                    )

                    replay_samples = [self.replay_buffer[i] for i in replay_idxs]

                    x_mem = torch.stack([s[0] for s in replay_samples]).to(self.device)
                    x_mark_mem = torch.stack([s[1] for s in replay_samples]).to(self.device)
                    y_mark_mem = torch.stack([s[3] for s in replay_samples]).to(self.device)

                    if self.replay_mode == "labels":
                        y_mem_true = torch.stack([s[2] for s in replay_samples]).to(self.device)
                        dec_inp_mem = torch.cat(
                            [y_mem_true[:, :label_len, :],
                             torch.zeros_like(y_mem_true[:, label_len:, :])],
                            dim=1
                        )
                        y_mem_now = self.model(x_mem, x_mark_mem, dec_inp_mem, y_mark_mem)
                        y_mem_now = y_mem_now[:, -pred_len:, :]
                        replay_loss = nn.functional.mse_loss(y_mem_now, y_mem_true)

                    elif self.replay_mode == "logits":
                        y_mem_logits = torch.stack([s[4] for s in replay_samples]).to(self.device)
                        dec_inp_mem = torch.zeros_like(y_mem_logits)
                        y_mem_now = self.model(x_mem, x_mark_mem, dec_inp_mem, y_mark_mem)
                        y_mem_now = y_mem_now[:, -pred_len:, :]
                        replay_loss = nn.functional.mse_loss(y_mem_now, y_mem_logits)

                    elif self.replay_mode == "both":
                        y_mem_true = torch.stack([s[2] for s in replay_samples]).to(self.device)
                        y_mem_logits = torch.stack([s[4] for s in replay_samples]).to(self.device)
                        dec_inp_mem = torch.cat(
                            [y_mem_true[:, :label_len, :],
                             torch.zeros_like(y_mem_true[:, label_len:, :])],
                            dim=1
                        )
                        y_mem_now = self.model(x_mem, x_mark_mem, dec_inp_mem, y_mark_mem)
                        y_mem_now = y_mem_now[:, -pred_len:, :]
                        replay_loss = (
                            self.beta * nn.functional.mse_loss(y_mem_now, y_mem_true) +
                            (1 - self.beta) * nn.functional.mse_loss(y_mem_now, y_mem_logits)
                        )
                    else:
                        raise ValueError("Unknown replay_mode")

                    loss = loss + self.alpha * replay_loss
                    replay_losses.append(replay_loss.item())

                # -------- backward --------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # -------- stocker en mémoire --------
                if self.replay_buffer_size > 0:
                    for xb, xb_mark, yb, yb_mark, ylog in zip(
                        x.cpu(), x_mark.cpu(), y.cpu(), y_mark.cpu(), preds_detached
                    ):
                        if self.replay_mode == "labels":
                            item = (xb, xb_mark, yb, yb_mark)
                        elif self.replay_mode == "logits":
                            item = (xb, xb_mark, None, yb_mark, ylog)
                        else:  # both
                            item = (xb, xb_mark, yb, yb_mark, ylog)
                        self.reservoir_sampling(self.replay_buffer, item)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"TaskLoss={np.mean(task_losses):.4f} | "
                f"ReplayLoss={np.mean(replay_losses) if replay_losses else 0:.4f}"
            )

        print(f"Replay buffer size: {len(self.replay_buffer)}")
