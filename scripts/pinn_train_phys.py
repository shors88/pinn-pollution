#!/usr/bin/env python
"""
Physics-informed PINN (advection–diffusion) tuned for Denver subset
──────────────────────────────────────────────────────────────────
• Lf = 20 Fourier features
• Huber β = 10
• λ_PDE ramp 0→50 over 400 epochs
• D unfrozen after 50 epochs
• Plume points (PM₂.₅ > 50 µg m⁻³) weighted ×50
• Logs every 50 epochs to results/hist_metrics.csv + curve PNG
"""
import math, argparse, pathlib, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt, pandas as pd
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

device = "cuda" if torch.cuda.is_available() else "cpu"
tt     = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)

# ─── Fourier features ────────────────────────────────────────────────────
def fourier_xy(xy, L=20):
    freqs = torch.logspace(0, math.log10(10), L, device=xy.device)
    proj  = 2 * math.pi * xy[..., None] * freqs
    return torch.cat([torch.sin(proj), torch.cos(proj)], -1).flatten(-2)

# ─── PINN model ───────────────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, hidden=128, depth=7, Lf=20, act="tanh", drop=0.1):
        super().__init__()
        actf = nn.Tanh if act == "tanh" else nn.ReLU
        layers, d = [], 4 * Lf + 3                          # 43 inputs
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
        return torch.expm1(f)                               # C = exp(f)−1 ≥ 0

# ─── PDE residual Ct + u∂xC + v∂yC − D∇²C = 0 ────────────────────────────
def pde_residual(model, x):
    x = x.clone().requires_grad_(True)
    C = model(x)
    g = torch.autograd.grad(C, x, torch.ones_like(C), create_graph=True)[0]
    dCdx, dCdy, dCdt = g[:,0:1], g[:,1:2], g[:,2:3]
    d2x = torch.autograd.grad(dCdx, x, torch.ones_like(dCdx),
                              create_graph=True)[0][:,0:1]
    d2y = torch.autograd.grad(dCdy, x, torch.ones_like(dCdy),
                              create_graph=True)[0][:,1:2]
    u, v = x[:,3:4], x[:,4:5]
    return (dCdt + u*dCdx + v*dCdy - model.D*(d2x+d2y)).square().mean()

# ─── training loop ────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",   type=int,   default=400)
    ap.add_argument("--activation",           default="tanh",
                    choices=["tanh","relu"])
    ap.add_argument("--resume",   default=None,
                    help="warm-start checkpoint (optional)")
    ap.add_argument("--nfreeze",  type=int,   default=50,
                    help="epochs to keep D fixed")
    ap.add_argument("--lr",       type=float, default=1e-4)
    ap.add_argument("--lam_max",  type=float, default=50.0,
                    help="λ ceiling for PDE term")
    args = ap.parse_args()

    # Load tensor
    d      = np.load("prep/train_tensor.npz")
    X_np   = d["X"].astype(np.float32)
    Y_np   = np.clip(d["Y"], 0.0, None).astype(np.float32).reshape(-1,1)
    hour   = d["t_idx"]; split = int(0.8 * (hour.max()+1))
    tr, te = hour < split, hour >= split
    X_tr, Y_tr = tt(X_np[tr]), tt(Y_np[tr])
    X_te, Y_te = tt(X_np[te]), tt(Y_np[te])
    print(f"TRAIN {len(X_tr):,}  TEST {len(X_te):,}")

    # Plume-weighted sampler ×50
    w_np = ((Y_np[tr] > 50).astype(np.float32).ravel()*49) + 1
    w    = torch.as_tensor(w_np, dtype=torch.float32, device=device)  # dtype fix
    loader = DataLoader(TensorDataset(X_tr, Y_tr),
                        batch_size=2048,
                        sampler=WeightedRandomSampler(w, len(w), replacement=True))

    # Model
    model = PINN(act=args.activation).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print("✓ resumed", args.resume)

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    huber = nn.SmoothL1Loss(beta=10.0)
    λ_D, D0 = 0.01, math.log(0.05)
    hist, lam_max = [], args.lam_max

    for ep in range(args.epochs+1):
        model.train(); opt.zero_grad()
        xb,yb = next(iter(loader))
        data_L = huber(model(xb), yb)
        pde_L  = pde_residual(model, X_tr[torch.randperm(len(X_tr))[:8000]])
        lam    = lam_max * ep / 400.0
        loss   = data_L + lam*pde_L + λ_D*(torch.log(model.D+1e-4)-D0).square()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()

        with torch.no_grad():
            if ep < args.nfreeze:
                model.D.copy_(torch.tensor(0.05, device=device))
            else:
                model.D.clamp_(0.02, 0.5)

        if ep % 50 == 0:
            model.eval()
            val_L = huber(model(X_te), Y_te).item()
            hist.append((ep, data_L.item(), pde_L.item(), val_L, model.D.item()))
            print(f"E{ep:04d} data={data_L:.3e}  PDE={pde_L:.3e}  "
                  f"λ={lam:5.1f}  val={val_L:.3e}  D={model.D.item():.3f}")

    # Save artefacts
    pathlib.Path("results").mkdir(exist_ok=True)
    tag = f"pinn_Lf20_lam{int(lam_max)}_b10"
    torch.save(model.state_dict(), f"results/ckpt_{tag}.pt")
    pd.DataFrame(hist, columns=["epoch","train_H","PDE","val_H","D"])\
      .to_csv("results/hist_metrics.csv", index=False)
    e,d,p,v,_D = zip(*hist)
    plt.semilogy(e,d,label="train Huber")
    plt.semilogy(e,p,label="PDE")
    plt.semilogy(e,v,label="val Huber")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.tight_layout(); plt.savefig(f"results/curve_{tag}.png", dpi=120)

if __name__ == "__main__":
    main()

