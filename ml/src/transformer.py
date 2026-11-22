"""Transformer encoder with multi-step direct forecasting head."""
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError("Input sequence exceeds configured positional encoding length")
        return x + self.pe[:, :x.size(1)]


class StockTransformer(nn.Module):
    def __init__(self, n_features, lookback=60, horizon=7, target_idx=3,
                 d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
                 dropout=.3, input_dropout=.1, noise_std=.02):
        super().__init__()
        if not (n_features >= 1 and lookback >= 1 and horizon >= 1 and 0 <= target_idx < n_features):
            raise ValueError("Invalid feature count, lookback, horizon, or target_idx")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.target_idx = target_idx
        self.lookback = lookback
        self.n_features = n_features
        self.noise_std = noise_std
        self.horizon = horizon
        # Close input must be in the target MinMaxScaler space (see preprocessing).
        self.linear_skip = nn.Linear(lookback, horizon)
        self.input_norm = nn.LayerNorm(n_features)
        self.input_dropout = nn.Dropout(input_dropout)
        self.input_projection = nn.Linear(n_features, d_model)
        self.position = PositionalEncoding(d_model, max_len=lookback)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
             dim_feedforward=dim_feedforward, dropout=dropout,
             batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.attn_score = nn.Linear(d_model, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, horizon))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.lookback, self.n_features):
            raise ValueError(f"Expected (batch, {self.lookback}, {self.n_features}), got {tuple(x.shape)}")
        last_close = x[:, -1, self.target_idx]
        skip = last_close.unsqueeze(1).expand(-1, self.horizon)
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        h = self.input_dropout(self.input_norm(x))
        h = self.position(self.input_projection(h))
        h = self.norm(self.encoder(h))
        weights = torch.softmax(self.attn_score(h), dim=1)
        pooled = (weights * h).sum(dim=1)
        return skip + self.head(pooled)
