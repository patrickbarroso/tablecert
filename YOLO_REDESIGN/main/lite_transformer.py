# lite_transformer.py
import torch
import torch.nn as nn

#Transformer Encoder Leve

class LiteTransformerBlock(nn.Module):
    def __init__(self, channels, nhead=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.proj_in(x)
        x_ = x.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        x_ = self.transformer(x_)
        x_ = x_.permute(0, 2, 1).reshape(b, c, h, w)
        return self.proj_out(x_ + x)
