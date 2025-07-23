#!/usr/bin/env python
"""
pinn_train.py  – DATA‑ONLY WARM‑START
---------------------------------------------------------
• Loads prep/train_tensor.npz created by build_tensor.py
• Uses only (x, y, t) columns   →   3‑input MLP
• Optimises MSE on PM2.5 without PDE terms
• Saves results/ckpt_dataonly.pt after training
"""

import numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt, pathlib

device = "cuda" if torch.cuda.is_available() else "cpu"
tt      = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
d      = np.load("prep/train_tensor.npz")
X_raw  = d["X"][:, :3]      # keep only x,y,t         ← IMPORTANT
Y_raw  = d["Y"]
t_idx  = d["t_idx"]

T_h    = t_idx.max() + 1
split  = int(0.8 * T_h)     # last 20 % of hours = validation
mask_v = t_idx >= split
mask_t = ~mask_v

X_tr, Y_tr = tt(X_raw[mask_t]), tt(Y_raw[mask_t])
X_va, Y_va = tt(X_raw[mask_v]), tt(Y_raw[mask_v])

print(f"TRAIN {len(X_tr)} pts, VAL {len(X_va)} pts (hold‑out hours ≥{split}/{T_h})")

# ── 2. NETWORK ───────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, hidden=64, depth=5):
        super().__init__()
        layers, fin = [], 3                 # 3 inputs (x,y,t)
        for _ in range(depth):
            layers += [nn.Linear(fin, hidden), nn.Tanh()]
            fin = hidden
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):          # guard against accidental NaNs
        return torch.nan_to_num(self.net(x), nan=0.0)

pinn  = MLP().to(device)
optim = torch.optim.Adam(pinn.parameters(), lr=3e-4)

# ── 3. TRAIN LOOP (data‑only) ────────────────────────────────────────────────
history = []
epochs  = 400          # ~2–3 min on CPU, <30 s on GPU

for ep in range(epochs + 1):
    pinn.train(); optim.zero_grad()
    pred = pinn(X_tr)
    train_L = ((pred - Y_tr)**2).mean()
    train_L.backward()
    optim.step()

    if ep % 50 == 0:
        pinn.eval()
        val_L = ((pinn(X_va) - Y_va)**2).mean().item()
        history.append((ep, train_L.item(), val_L))
        print(f"E{ep:04d}  train={train_L.item():.4e}  val={val_L:.4e}")

# ── 4. PLOT & SAVE ───────────────────────────────────────────────────────────
if history:
    e, tr, va = zip(*history)
    plt.semilogy(e, tr, label="train")
    plt.semilogy(e, va, label="val")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.tight_layout()
    pathlib.Path("results").mkdir(exist_ok=True)
    plt.savefig("results/dataonly_curve.png", dpi=150)
    plt.show()
    torch.save(pinn.state_dict(), "results/ckpt_dataonly.pt")
    print("✓ saved → results/ckpt_dataonly.pt")
else:
    print("⚠️  No points logged (check data).")

