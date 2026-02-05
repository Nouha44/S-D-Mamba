import torch
import numpy as np
import random
from torch import nn


class DERContinualSMamba:
    """
    Implémentation STRICTE de DER et DER++
    pour S-Mamba (seq2seq)
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
        replay_mode="labels",  # labels | logits | both
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

    # --------------------------------------------------
    # Reservoir sampling (standard, correct)
    # --------------------------------------------------
    def reservoir_sampling(self, new_item):
        if self.replay_buffer_size == 0:
            return
        if len(self.replay_buffer) < self.replay_buffer_size:
            self.replay_buffer.append(new_item)
        else:
            j = random.randint(0, len(self.replay_buffer))
            if j < self.replay_buffer_size:
                self.replay_buffer[j] = new_item

    # --------------------------------------------------
    # Train one task
    # --------------------------------------------------
    def fit_one_task(self, train_loader, label_len, pred_len, task_idx=0, epochs=5):

        self.model.train()

        for epoch in range(epochs):
            task_losses, replay_losses = [], []

            for x, x_mark, y, y_mark in train_loader:
                x = x.to(self.device)
                x_mark = x_mark.to(self.device)
                y = y.to(self.device)
                y_mark = y_mark.to(self.device)

                # ---- decoder input (PART OF INPUT) ----
                dec_inp = torch.cat(
                    [y[:, :label_len, :],
                     torch.zeros_like(y[:, label_len:, :])],
                    dim=1
                )

                # ================= CURRENT TASK =================
                preds = self.model(x, x_mark, dec_inp, y_mark)
                preds = preds[:, -pred_len:, :]
                loss = self.criterion(preds, y)
                task_losses.append(loss.item())

                preds_detached = preds.detach().cpu()

                # ================= REPLAY =================
                if task_idx > 0 and len(self.replay_buffer) > 0:
                    replay_idx = np.random.choice(
                        len(self.replay_buffer),
                        min(len(self.replay_buffer), x.size(0)),
                        replace=False
                    )

                    samples = [self.replay_buffer[i] for i in replay_idx]

                    x_m = torch.stack([s["x"] for s in samples]).to(self.device)
                    x_mark_m = torch.stack([s["x_mark"] for s in samples]).to(self.device)
                    dec_inp_m = torch.stack([s["dec_inp"] for s in samples]).to(self.device)
                    y_mark_m = torch.stack([s["y_mark"] for s in samples]).to(self.device)

                    preds_now = self.model(x_m, x_mark_m, dec_inp_m, y_mark_m)
                    preds_now = preds_now[:, -pred_len:, :]

                    if self.replay_mode == "labels":
                        y_true = torch.stack([s["y"] for s in samples]).to(self.device)
                        replay_loss = nn.functional.mse_loss(preds_now, y_true)

                    elif self.replay_mode == "logits":
                        old_logits = torch.stack([s["logits"] for s in samples]).to(self.device)
                        replay_loss = nn.functional.mse_loss(preds_now, old_logits)

                    elif self.replay_mode == "both":
                        y_true = torch.stack([s["y"] for s in samples]).to(self.device)
                        old_logits = torch.stack([s["logits"] for s in samples]).to(self.device)
                        replay_loss = (
                            self.beta * nn.functional.mse_loss(preds_now, y_true) +
                            (1 - self.beta) * nn.functional.mse_loss(preds_now, old_logits)
                        )
                    else:
                        raise ValueError("Unknown replay_mode")

                    loss = loss + self.alpha * replay_loss
                    replay_losses.append(replay_loss.item())

                # ================= BACKWARD =================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # ================= STORE MEMORY =================
                for xb, xb_m, yb, yb_m, dinp, logit in zip(
                    x.cpu(), x_mark.cpu(), y.cpu(), y_mark.cpu(),
                    dec_inp.cpu(), preds_detached
                ):
                    item = {
                        "x": xb,
                        "x_mark": xb_m,
                        "dec_inp": dinp,
                        "y_mark": yb_m,
                        "y": yb,
                        "logits": logit,
                    }
                    self.reservoir_sampling(item)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"TaskLoss={np.mean(task_losses):.4f} | "
                f"ReplayLoss={np.mean(replay_losses) if replay_losses else 0:.4f}"
            )

        print(f"Replay buffer size: {len(self.replay_buffer)}")
