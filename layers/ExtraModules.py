class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        # x: [B, L, N]
        x_perm = x.permute(0, 2, 1)  # B N L
        trend = self.moving_avg(x_perm).permute(0, 2, 1)  # B L N
        residual = x - trend
        return trend, residual

class CrossVariableAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # x: [B, L, N] -> treat N as tokens
        x_perm = x.permute(0, 2, 1)  # B N L
        attn_out, _ = self.attn(x_perm, x_perm, x_perm)
        return attn_out.permute(0, 2, 1)  # B L N
