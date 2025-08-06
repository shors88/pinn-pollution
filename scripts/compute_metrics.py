#!/usr/bin/env python
"""
compute_metrics.py — robust evaluation for PINN checkpoints
"""

import argparse, math, warnings, numpy as np, torch, torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=DeprecationWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
tt     = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)

# Fourier features ----------------------------------------------------------
def fourier_xy(xy, L):
    f = torch.logspace(0, math.log10(10), L, device=xy.device)
    p = 2 * math.pi * xy[..., None] * f
    return torch.cat([torch.sin(p), torch.cos(p)], -1).flatten(-2)

# PINN (matches training topology) -----------------------------------------
class PINN(nn.Module):
    def __init__(self, Lf=20, hidden=128, depth=7, act="tanh", drop=0.1):
        super().__init__()
        actf = nn.Tanh if act == "tanh" else nn.ReLU
        layers, d = [], 4 * Lf + 3
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), actf(), nn.Dropout(drop)]
            d = hidden
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.D   = nn.Parameter(torch.tensor(0.05))
        self.Lf  = Lf
    def forward(self, x):
        enc = fourier_xy(x[:, :2], self.Lf)
        z   = torch.cat([enc, x[:, 2:]], -1)
        f   = torch.clamp(self.net(z), -15., 5.)
        return torch.expm1(f)

# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--Lf",       type=int,   default=20)
    ap.add_argument("--depth",    type=int,   default=7)
    ap.add_argument("--dropout",  type=float, default=0.1)
    ap.add_argument("--activation", default="tanh", choices=["tanh","relu"])
    args = ap.parse_args()

    # load design tensor ----------------------------------------------------
    d      = np.load(args.tensor)
    X_np   = d["X"].astype(np.float32)
    Y_np   = np.clip(d["Y"], 0.0, None).astype(np.float32).reshape(-1,1)
    hour   = d["t_idx"]; split = int(0.8 * (hour.max()+1))
    mask   = hour >= split
    X_te   = tt(X_np[mask])
    Y_te   = Y_np[mask]                 # still numpy here

    # instantiate model -----------------------------------------------------
    model = PINN(Lf=args.Lf, depth=args.depth,
                 act=args.activation, drop=args.dropout).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    with torch.no_grad():
        Y_hat = model(X_te).cpu().numpy()

    # ensure plain float32 ndarrays -----------------------------------------
    Y_true = np.asarray(Y_te,  dtype=np.float32).ravel()
    Y_pred = np.asarray(Y_hat, dtype=np.float32).ravel()

    # remove any NaNs (shouldn’t happen but be safe)
    good   = ~np.isnan(Y_pred)
    Y_true, Y_pred = Y_true[good], Y_pred[good]

    # metrics ---------------------------------------------------------------
    mae  = float(np.mean(np.abs(Y_pred - Y_true)))
    rmse = float(np.sqrt(np.mean((Y_pred - Y_true) ** 2)))
    r2   = r2_score(Y_true, Y_pred)

    print("\nHeld-out performance (last 20 % hours):")
    print(f"  MAE  = {mae:7.3f} µg m⁻³")
    print(f"  RMSE = {rmse:7.3f} µg m⁻³")
    print(f"  R²   = {r2:7.3f}")

if __name__ == "__main__":
    main()

