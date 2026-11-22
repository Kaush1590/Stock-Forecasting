"""Reproducible model training with early stopping."""
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can fail on unsupported operations; reproducibility
    # is best-effort across hardware, drivers and PyTorch versions.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, X_train, y_train, X_val, y_val, epochs=80,
                batch_size=32, learning_rate=1e-3, patience=12, device="cpu", seed=42):
    if min(len(X_train), len(y_train), len(X_val), len(y_val)) == 0:
        raise ValueError("Training and validation datasets must not be empty")
    if len(X_train) != len(y_train) or len(X_val) != len(y_val):
        raise ValueError("Features and labels must have matching batch lengths")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or patience < 1:
        raise ValueError("Invalid training hyperparameters")
    set_seed(seed)
    train_ds = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                             torch.as_tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.as_tensor(X_val, dtype=torch.float32),
                           torch.as_tensor(y_val, dtype=torch.float32))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=4)
    criterion = nn.HuberLoss()
    best_loss, best_state, wait = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            optimizer.step()
            train_total += loss.item() * len(xb)
        model.eval()
        val_total = 0.
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_total += criterion(model(xb), yb).item() * len(xb)
        train_loss = train_total / len(train_ds)
        val_loss = val_total / len(val_ds)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:03d}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f}")
        if val_loss < best_loss:
            best_loss, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
