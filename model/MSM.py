import torch
import torch.nn as nn
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted
from layers.ExtraModules import SeriesDecomp
from mamba_ssm import Mamba
class CrossVariableAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):  # x: [B, L, N]
        # Transpose to [B, N, L] to treat variables as tokens
        x = x.transpose(1, 2)
        B, N, L = x.shape
        qkv = self.qkv(x)  # [B, N, 3*d_model]
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.bmm(q, k.transpose(1, 2)) / (L ** 0.5)
        attn = self.softmax(attn)
        out = torch.bmm(attn, v)
        out = self.proj(out)
        # Back to [B, L, N]
        return out.transpose(1, 2)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model,
                                                    configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder([...])  # keep your existing encoder layers
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # NEW blocks
        self.decomp = series_decomp(kernel_size=25)
        self.cross_att = CrossVariableAttention(d_model=configs.d_model)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Decompose
        trend, residual = self.decomp(enc_out)  # [B, L, N]

        # Optional: SSM or encoder processing separately
        trend, _ = self.encoder(trend)
        residual, _ = self.encoder(residual)

        # Cross-variable attention
        trend_att = self.cross_att(trend)
        res_att = self.cross_att(residual)
        enc_out = trend_att + res_att

        # Projection
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :x_enc.shape[2]]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
