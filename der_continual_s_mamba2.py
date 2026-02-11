import torch
import numpy as np
import random
from torch import nn
from samformer.samformer import SAMFormerArchitecture


class DERContinualSMamba:
    """
    Dark Experience Replay (DER / DER++) pour forecasting avec S-Mamba
    Implémentation ALIGNÉE avec DERContinualSAMFormer
    """

    def __init__(self, model, optimizer, criterion, device,
                 replay_buffer_size=500, alpha=1.0, beta=0.5,
                 replay_mode="labels"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.replay_buffer_size = replay_buffer_size
        self.alpha = alpha
        self.beta = beta
        self.replay_mode = replay_mode

        # buffers (IDENTIQUE à SAMFormer)
        self.x_buffer = None  # encoder input
        self.x_mark_buffer = None
        self.dec_inp_buffer = None
        self.y_mark_buffer = None
        self.y_buffer = None
        self.logits_buffer = None
        self.task_id_buffer = None

        self.buffer_filled = 0
        self.n_seen_examples = 0

    # --------------------------------------------------
    # Reservoir Sampling (STRICTEMENT IDENTIQUE)
    # --------------------------------------------------
    def _add_to_buffer(self, x, x_mark, dec_inp, y_mark, y=None, logits=None, task_id=None):
        batch_size = x.shape[0]

        if self.x_buffer is None:
            self.x_buffer = torch.zeros(
                (self.replay_buffer_size, *x.shape[1:]), dtype=torch.float32
            )
            self.x_mark_buffer = torch.zeros(
                (self.replay_buffer_size, *x_mark.shape[1:]), dtype=torch.float32
            )
            self.dec_inp_buffer = torch.zeros(
                (self.replay_buffer_size, *dec_inp.shape[1:]), dtype=torch.float32
            )
            self.y_mark_buffer = torch.zeros(
                (self.replay_buffer_size, *y_mark.shape[1:]), dtype=torch.float32
            )
            if y is not None:
                self.y_buffer = torch.zeros(
                    (self.replay_buffer_size, *y.shape[1:]), dtype=torch.float32
                )
            if logits is not None:
                self.logits_buffer = torch.zeros(
                    (self.replay_buffer_size, *logits.shape[1:]), dtype=torch.float32
                )
            self.task_id_buffer = torch.zeros(
                (self.replay_buffer_size,), dtype=torch.long
            )

        for i in range(batch_size):
            self.n_seen_examples += 1
            if self.buffer_filled < self.replay_buffer_size:
                idx = self.buffer_filled
                self.buffer_filled += 1
            else:
                idx = random.randint(0, self.n_seen_examples - 1)
                if idx >= self.replay_buffer_size:
                    continue

            self.x_buffer[idx] = x[i].detach().cpu()
            self.x_mark_buffer[idx] = x_mark[i].detach().cpu()
            self.dec_inp_buffer[idx] = dec_inp[i].detach().cpu()
            self.y_mark_buffer[idx] = y_mark[i].detach().cpu()

            if y is not None:
                self.y_buffer[idx] = y[i].detach().cpu()
            if logits is not None:
                self.logits_buffer[idx] = logits[i].detach().cpu()
            self.task_id_buffer[idx] = task_id

    # --------------------------------------------------
    # Sample buffer (IDENTIQUE)
    # --------------------------------------------------
    def _sample_buffer(self, batch_size):
        if self.buffer_filled == 0:
            return None

        indices = torch.randint(0, self.buffer_filled, (batch_size,))
        return (
            self.x_buffer[indices].to(self.device),
            self.x_mark_buffer[indices].to(self.device),
            self.dec_inp_buffer[indices].to(self.device),
            self.y_mark_buffer[indices].to(self.device),
            self.y_buffer[indices].to(self.device) if self.y_buffer is not None else None,
            self.logits_buffer[indices].to(self.device) if self.logits_buffer is not None else None,
            self.task_id_buffer[indices].to(self.device),
        )

    # --------------------------------------------------
    # Fit one task (ALIGNÉ SAMFormer)
    # --------------------------------------------------
    def fit_one_task(self, train_loader, label_len, pred_len, task_idx=0, epochs=5):
        self.model.train()

        for epoch in range(epochs):
            task_losses, replay_losses = []

            for x, x_mark, y, y_mark in train_loader:
                x = x.to(self.device)
                x_mark = x_mark.to(self.device)
                y = y.to(self.device)
                y_mark = y_mark.to(self.device)

                # decoder input
                dec_inp = torch.cat(
                    [y[:, :label_len, :], torch.zeros_like(y[:, label_len:, :])], dim=1
                )

                # ---------- CURRENT TASK ----------
                preds = self.model(x, x_mark, dec_inp, y_mark)
                preds = preds[:, -pred_len:, :]
                loss = self.criterion(preds, y)
                task_losses.append(loss.item())
                preds_detached = preds.detach()

                # ---------- REPLAY ----------
                if self.buffer_filled > 0:
                    x_m, x_mark_m, dec_m, y_mark_m, y_m, logits_m, _ = \
                        self._sample_buffer(min(x.size(0), self.buffer_filled))

                    preds_mem = self.model(x_m, x_mark_m, dec_m, y_mark_m)
                    preds_mem = preds_mem[:, -pred_len:, :]

                    if self.replay_mode == "labels":
                        replay_loss = nn.functional.mse_loss(preds_mem, y_m)
                    elif self.replay_mode == "logits":
                        replay_loss = nn.functional.mse_loss(preds_mem, logits_m)
                    elif self.replay_mode == "both":
                        replay_loss = (
                            self.beta * nn.functional.mse_loss(preds_mem, y_m)
                            + (1 - self.beta) * nn.functional.mse_loss(preds_mem, logits_m)
                        )
                    else:
                        replay_loss = 0.0

                    loss = loss + self.alpha * replay_loss
                    replay_losses.append(replay_loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(
                f"Epoch {epoch+1}/{epochs} "
                f"TaskLoss={np.mean(task_losses):.4f} "
                f"ReplayLoss={np.mean(replay_losses) if replay_losses else 0:.4f}"
            )

        # ---------- FILL BUFFER AFTER TASK (IDENTIQUE SAMFormer) ----------
        with torch.no_grad():
            for x, x_mark, y, y_mark in train_loader:
                dec_inp = torch.cat(
                    [y[:, :label_len, :], torch.zeros_like(y[:, label_len:, :])], dim=1
                )
                preds = self.model(x.to(self.device), x_mark.to(self.device),
                                   dec_inp.to(self.device), y_mark.to(self.device))
                preds = preds[:, -pred_len:, :]

                if self.replay_mode == "labels":
                    self._add_to_buffer(x, x_mark, dec_inp, y_mark, y=y, task_id=task_idx)
                elif self.replay_mode == "logits":
                    self._add_to_buffer(x, x_mark, dec_inp, y_mark, logits=preds, task_id=task_idx)
                elif self.replay_mode == "both":
                    self._add_to_buffer(x, x_mark, dec_inp, y_mark, y=y, logits=preds, task_id=task_idx)

        print(f"Replay buffer size after task {task_idx+1}: {self.buffer_filled}")
