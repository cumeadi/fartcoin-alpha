"""
LSTM Meta-Model — Fartcoin Alpha Framework

Replaces LightGBM meta-model. Treats META_FEATURES as a time sequence
(LOOKBACK hourly bars) so the model can learn regime transitions rather
than scoring each bar independently.

Architecture: 2-layer LSTM → Dropout → Linear(hidden→1) → Sigmoid
Training:     BCELoss, Adam, ReduceLROnPlateau, early stopping, grad clipping
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

LOOKBACK = 10   # sequence length (hours)

_DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)


class TradingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq, features)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.drop(out[:, -1, :])   # last timestep only
        return torch.sigmoid(self.fc(out))


def build_sequences(features_array, targets_array, lookback=LOOKBACK):
    """
    Convert flat (N, F) array into sequences (N-lookback, lookback, F) and
    corresponding targets (N-lookback,).
    """
    X, y = [], []
    for i in range(lookback, len(features_array)):
        X.append(features_array[i - lookback: i])
        y.append(targets_array[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_size: int,
    hidden_size: int = 64,
    num_layers: int  = 2,
    dropout: float   = 0.2,
    lr: float        = 0.001,
    epochs: int      = 100,
    patience: int    = 10,
    batch_size: int  = 64,
    val_frac: float  = 0.15,
    seed: int        = 42,
) -> TradingLSTM:
    """
    Train a TradingLSTM on pre-built sequence arrays.

    X_train: (N, LOOKBACK, n_features)
    y_train: (N,) binary 0/1

    Returns best-weight model (restored on early stop).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # chronological train/val split (no shuffle — time series)
    n_val   = max(1, int(len(X_train) * val_frac))
    n_tr    = len(X_train) - n_val
    X_tr, X_val = X_train[:n_tr], X_train[n_tr:]
    y_tr, y_val = y_train[:n_tr], y_train[n_tr:]

    # class weights to handle imbalance
    pos_rate  = y_tr.mean() + 1e-9
    pos_weight = torch.tensor([(1.0 - pos_rate) / pos_rate], device=_DEVICE)

    tr_ds  = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    tr_dl  = DataLoader(tr_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model     = TradingLSTM(input_size, hidden_size, num_layers, dropout).to(_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.BCELoss(reduction="mean")

    best_val  = float("inf")
    best_wts  = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(epochs):
        # ── train ──────────────────────────────────────────────────────
        model.train()
        for Xb, yb in tr_dl:
            Xb, yb = Xb.to(_DEVICE), yb.to(_DEVICE)
            optimizer.zero_grad()
            pred = model(Xb).squeeze(1)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ── validate ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_dl:
                Xb, yb = Xb.to(_DEVICE), yb.to(_DEVICE)
                pred = model(Xb).squeeze(1)
                val_loss += criterion(pred, yb).item()
        val_loss /= len(val_dl)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val  = val_loss
            best_wts  = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_wts)
    return model


def predict_lstm(model: TradingLSTM, X_seq: np.ndarray) -> float:
    """
    Score a single sequence.
    X_seq: (LOOKBACK, n_features) — will be promoted to (1, LOOKBACK, n_features).
    Returns probability [0, 1].
    """
    model.eval()
    x = torch.from_numpy(X_seq.astype(np.float32)).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        prob = float(model(x).squeeze().item())
    return prob
