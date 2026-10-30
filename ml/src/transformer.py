import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, maxLen: int = 50000):
        super().__init__()

        zeroes = torch.zeros(maxLen, d_model)
        position = torch.arange(0, maxLen, dtype=torch.int32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        zeroes[:, 0::2] = torch.sin(position * div)
        zeroes[:, 1::2] = torch.cos(position * div[:zeroes[:, 1::2].shape[1]])

        self.register_buffer("zeroes", zeroes.unsqueeze(0))

    def forward(self, x):
        return x + self.zeroes[:, :x.size(1)]

class StockTransformer(nn.Module):
    def __init__(self, n_features, horizon=7, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()

        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")

        self.input_projection = nn.Linear(n_features, d_model)
        self.position = PositionalEncoding(d_model)

        layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(layers, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.position(x)
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])
        return self.head(x)
