import torch
import numpy as np
import random
from torch import nn

class DERContinualSMamba:
    """
    Dark Experience Replay (DER / DER++) for forecasting with S-Mamba,
    implemented exactly like DERContinualSAMFormer.
    """

    def __init__(self, model, learning_rate=1e-3, batch_size=64, num_epochs=5,
                 replay_buffer_size=500, alpha=1.0, beta=0.5,
                 replay_mode="labels", device="cpu"):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.replay_buffer_size = replay_buffer_size
        self.alpha = alpha
        self.beta = beta
        self.replay_mode = replay_mode
        self.device = device

        # Buffers
        self.x_buffer = None
        self.x_mark_buffer = None
        self.dec_inp_buffer = None
        self.y_mark_buffer = None

        self.y_buffer = None
        self.logits_buffer = None
        self.task_id_buffer = None

        self.buffer_filled = 0
        self.n_seen_examples = 0

    # ---------------- Reservoir Sampling ----------------
    def _add_to_buffer(self, x, x_mark, dec_inp, y_mark, y=None, logits=None, task_id=None):
        batch_size = x.shape[0]

        if self.x_buffer is None:
            self.x_buffer = torch.zeros((self.replay_buffer_size, *x.shape[1:]), dtype=torch.float32)
            self.x_mark_buffer = torch.zeros((self.replay_buffer_size, *x_mark.shape[1:]), dtype=torch.float32)
            self.dec_inp_buffer = torch.zeros((self.replay_buffer_size, *dec_inp.shape[1:]), dtype=torch.float32)
            self.y_mark_buffer = torch.zeros((self.replay_buffer_size, *y_mark.shape[1:]), dtype=torch.float32)

            if y is not None:
                self.y_buffer = torch.zeros((self.replay_buffer_size, *y.shape[1:]), dtype=torch.float32)
            if logits is not None:
                self.logits_buffer = torch.zeros((self.replay_buffer_size, *logits.shape[1:]), dtype=torch.float32)

            self.task_id_buffer = torch.zeros((self.replay_buffer_size,), dtype=torch.long)

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

    # ---------------- Sample from buffer ----------------
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
            self.task_id_buffer[indices].to(self.device)
        )

    # ---------------- Fit one task ----------------
    def fit_one_task(self, x, x_mark, y, y_mark, optimizer, criterion, task_idx=0):
        self.model.train()

        # ---------------- Convert to tensor ----------------
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x_mark = torch.tensor(x_mark, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        y_mark = torch.tensor(y_mark, dtype=torch.float32, device=self.device)

        # ---------------- Decoder input ----------------
        dec_inp = torch.cat([y[:, :y_mark.shape[1], :],
                             torch.zeros_like(y[:, y_mark.shape[1]:, :])], dim=1)

        for epoch in range(self.num_epochs):
            perm = torch.randperm(x.shape[0])
            task_losses, replay_losses = [], []

            for i in range(0, x.shape[0], self.batch_size):
                idxs = perm[i:i + self.batch_size]
                x_batch, x_mark_batch = x[idxs], x_mark[idxs]
                y_batch, y_mark_batch = y[idxs], y_mark[idxs]
                dec_batch = dec_inp[idxs]

                # ---------- CURRENT TASK ----------
                out = self.model(x_batch, x_mark_batch, dec_batch, y_mark_batch)
                loss = criterion(out, y_batch)
                task_losses.append(loss.item())

                # ---------- REPLAY ----------
                if self.buffer_filled > 0:
                    x_mem, x_mark_mem, dec_mem, y_mark_mem, y_mem, logits_mem, _ = \
                        self._sample_buffer(min(self.batch_size, self.buffer_filled))

                    out_mem = self.model(x_mem, x_mark_mem, dec_mem, y_mark_mem)

                    if self.replay_mode == "labels" and y_mem is not None:
                        replay_loss = nn.functional.mse_loss(out_mem, y_mem)
                    elif self.replay_mode == "logits" and logits_mem is not None:
                        replay_loss = nn.functional.mse_loss(out_mem, logits_mem)
                    elif self.replay_mode == "both":
                        replay_loss = (self.beta * nn.functional.mse_loss(out_mem, y_mem) +
                                       (1 - self.beta) * nn.functional.mse_loss(out_mem, logits_mem))
                    else:
                        replay_loss = 0.0

                    loss = loss + self.alpha * replay_loss
                    replay_losses.append(replay_loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{self.num_epochs} "
                  f"TaskLoss={np.mean(task_losses):.4f} "
                  f"ReplayLoss={np.mean(replay_losses) if replay_losses else 0:.4f}")

        # ---------- FILL BUFFER AFTER TASK ----------
        with torch.no_grad():
            self._add_to_buffer(x, x_mark, dec_inp, y_mark,
                                y=y, logits=out, task_id=task_idx)

        print(f"Replay buffer size after task {task_idx+1}: {self.buffer_filled}")
