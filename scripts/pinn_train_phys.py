#!/usr/bin/env python
"""
Resume‑and‑freeze PINN (advection‑diffusion, plume‑weighted)
────────────────────────────────────────────────────────────
 • --resume ckpt    : warm‑start from previous training
 • --epochs N       : extra epochs to run (default 250)
 • --lr LR          : Adam learning‑rate (default 1e‑4)
 • --nfreeze K      : keep D frozen for K epochs after resume
 • λ_PDE ramps 0→500 over first 400 epochs (immediate start)
 • Huber β = 20 µg, plume samples weighted ×50
"""

import math, argparse, pathlib, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

device = "cuda" if torch.cuda.is_available() else "cpu"
tt = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)

# ───────── Fourier features ────────────────────────────────────────────────
def fourier_xy(xy, L=10):
    freqs = torch.logspace(0, math.log10(10), L, device=xy.device)
    proj  = 2 * math.pi * xy[..., None] * freqs
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1).flatten(-2)

# ───────── PINN model ──────────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, hidden=128, depth=7, dropout=0.1, Lf=10, act="tanh"):
        super().__init__()
        act_fn = nn.Tanh if act == "tanh" else nn.ReLU
        in_dim = 4 * Lf + 3                     # [sin,cos] *2*Lf  +  t,u,v
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), act_fn(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        self.D   = nn.Parameter(torch.tensor(0.05))
        self.Lf  = Lf

    def forward(self, x, return_log=False):
        xy_enc = fourier_xy(x[:, :2], self.Lf)  # lon,lat → Fourier
        rest   = x[:, 2:]                       # t,u,v
        f      = self.net(torch.cat([xy_enc, rest], dim=-1))
        f      = torch.clamp(torch.nan_to_num(f), -15.0, 5.0)
        return f if return_log else torch.expm1(f)   # C = e^f − 1

# ───────── PDE residual ‑‑ Ct + u∇x + v∇y − D∇²C = 0 ──────────────────────
def pde_residual(model, x):
    x = x.clone().requires_grad_(True)
    C = model(x)
    g = torch.autograd.grad(C, x, torch.ones_like(C), create_graph=True)[0]
    dCdx, dCdy, dCdt = g[:,0:1], g[:,1:2], g[:,2:3]
    d2Cdx2 = torch.autograd.grad(dCdx, x, torch.ones_like(dCdx),
                                 create_graph=True)[0][:, 0:1]
    d2Cdy2 = torch.autograd.grad(dCdy, x, torch.ones_like(dCdy),
                                 create_graph=True)[0][:, 1:2]
    u, v = x[:,3:4], x[:,4:5]
    return (dCdt + u*dCdx + v*dCdy - model.D*(d2Cdx2+d2Cdy2)).square().mean()

# ───────── training loop ───────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--activation", default="tanh", choices=["tanh","relu"])
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--nfreeze", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    # -------- load data ----------------------------------------------------
    d       = np.load("prep/train_tensor.npz")
    X_raw   = d["X"]
    Y_lin   = np.clip(d["Y"].astype(np.float32), 0.0, None).reshape(-1, 1)
    hour    = d["t_idx"]
    split   = int(0.8 * (hour.max() + 1))
    tr, te  = hour < split, hour >= split
    X_tr, Y_tr = tt(X_raw[tr]), tt(Y_lin[tr])
    X_te, Y_te = tt(X_raw[te]), tt(Y_lin[te])
    print(f"TRAIN {len(X_tr):,}  TEST {len(X_te):,}  (split hr {split}/{hour.max()+1})")

    # -------- plume weighting ×50 -----------------------------------------
    w_np   = ((Y_lin[tr] > 50).astype(np.float32).ravel() * 49.0) + 1.0
    w_torch = torch.tensor(w_np, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tr, Y_tr),
                        batch_size=2048,
                        sampler=WeightedRandomSampler(w_torch,
                                                      num_samples=len(w_torch),
                                                      replacement=True))

    # -------- model --------------------------------------------------------
    model = PINN(act=args.activation).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"✓ resumed from {args.resume}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    huber = nn.SmoothL1Loss(beta=20.0)
    λ_D   = 0.01;   D_prior = math.log(0.05)

    hist = []
    for ep in range(args.epochs + 1):
        model.train(); optim.zero_grad()

        xb, yb = next(iter(loader))
        data_L = huber(model(xb), yb)

        col   = X_tr[torch.randperm(len(X_tr))[:10_000]]
        pde_L = pde_residual(model, col)

        lam = 500.0 * ep / 400.0           # immediate λ ramp (0→500 in 400 ep)
        D_reg = λ_D * (torch.log(model.D + 1e-4) - D_prior).square()
        loss  = data_L + lam * pde_L + D_reg
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()

        # freeze / clamp diffusivity
        with torch.no_grad():
            if ep < args.nfreeze:
                model.D.copy_(torch.tensor(0.05, device=device))
            else:
                model.D.clamp_(0.02, 0.5)

        # log
        if ep % 50 == 0:
            model.eval()
            test_L = huber(model(X_te), Y_te).item()
            hist.append((ep, data_L.item(), pde_L.item(), test_L))
            print(f"E{ep:04d}  data={data_L:.3e}  PDE={pde_L:.3e}  "
                  f"λ={lam:6.1f}  test={test_L:.3e}  D={model.D.item():.3f}")

    # -------- save & curves ----------------------------------------------
    pathlib.Path("results").mkdir(exist_ok=True)
    tag = f"plumePINN_freezeD_a{args.activation}"
    torch.save(model.state_dict(), f"results/ckpt_{tag}.pt")
    print(f"✓ saved → results/ckpt_{tag}.pt")

    if hist:
        e, d, p, v = zip(*hist)
        plt.semilogy(e, d, label="train Huber")
        plt.semilogy(e, p, label="PDE")
        plt.semilogy(e, v, label="test Huber")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
        plt.tight_layout(); plt.savefig(f"results/curve_{tag}.png", dpi=120)
        plt.show()

if __name__ == "__main__":
    main()
