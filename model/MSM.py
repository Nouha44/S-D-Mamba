import torch
import torch.nn as nn
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted
from mamba_ssm import Mamba

# ----------------------------
# Series Decomposition Block
# ----------------------------
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        # x: [B, L, N]
        x_perm = x.permute(0, 2, 1)  # B N L
        trend = self.moving_avg(x_perm).permute(0, 2, 1)  # B L N
        residual = x - trend
        return trend, residual

# ----------------------------
# Cross-Variable Attention
# ----------------------------
class CrossVariableAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # x: [B, L, D] -> treat L as sequence, D as embedding dim
        attn_out, _ = self.attn(x, x, x)
        return attn_out  # [B, L, D]

# ----------------------------
# MSM Model
# ----------------------------
class Model(nn.Module):
    def __init__(self, configs, num_vars=1):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.num_vars = num_vars  # N

        # ----------------------------
        # Embedding
        # ----------------------------
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model,
                                                    configs.embed, configs.freq, configs.dropout)

        # ----------------------------
        # Encoder Layers
        # ----------------------------
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=2, expand=1),
                    Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=2, expand=1),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # ----------------------------
        # Series Decomposition + Cross-Variable Attention
        # ----------------------------
        self.decomp = SeriesDecomp(kernel_size=25)

        # Linear projection N -> d_model before attention
        self.proj_in = nn.Linear(self.num_vars, configs.d_model)

        # Cross-variable attention
        self.cross_att = CrossVariableAttention(d_model=configs.d_model, num_heads=4)

        # Linear projection back d_model -> N after attention
        self.proj_out = nn.Linear(configs.d_model, self.num_vars)

    # ----------------------------
    # Forecast Function
    # ----------------------------
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # B L N -> B L N (or B L D if embedding outputs D)

        # ----------------------------
        # Series Decomposition
        # ----------------------------
        trend, residual = self.decomp(enc_out)  # B L N

        # ----------------------------
        # Project to d_model
        # ----------------------------
        trend = self.proj_in(trend)       # B L D
        residual = self.proj_in(residual) # B L D

        # ----------------------------
        # Encoder Processing
        # ----------------------------
        trend, _ = self.encoder(trend)
        residual, _ = self.encoder(residual)

        # ----------------------------
        # Cross-Variable Attention
        # ----------------------------
        trend_att = self.cross_att(trend)
        residual_att = self.cross_att(residual)

        enc_out = trend_att + residual_att  # combine components

        # ----------------------------
        # Project back to original variable dimension
        # ----------------------------
        enc_out = self.proj_out(enc_out)  # B L N

        # ----------------------------
        # Final projection to prediction length
        # ----------------------------
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :self.num_vars]  # B S N

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)

        return dec_out

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, N]
