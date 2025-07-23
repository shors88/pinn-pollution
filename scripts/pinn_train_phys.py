#!/usr/bin/env python
"""
Physics‑informed PINN: advection‑diffusion with winds (u10, v10)

• Uses train_tensor.npz built by build_tensor.py  (columns: lon, lat, t, u10, v10)
• Holds out the last 20 % of hours for validation
• Loss = data_MSE  +  λ_pde * PDE‑residual  (ramps from 0→10)
"""

import numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt, pathlib

device = "cuda" if torch.cuda.is_available() else "cpu"
tt = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)

# ╔════════ 1. LOAD DATA ══════════════════════════════════════════════════════╗
d      = np.load("prep/train_tensor.npz")
X_all  = tt(d["X"])          # (N,5)
Y_all  = tt(d["Y"])          # (N,1)
hour   = d["t_idx"]          # (N,) integer 0…H−1

H       = hour.max() + 1
hold_hr = int(0.8 * H)       # hours < hold_hr → train, else val
hold_np = hour >= hold_hr

train_mask_t = torch.tensor(~hold_np, dtype=torch.bool, device=device)
val_mask_t   = torch.tensor( hold_np, dtype=torch.bool, device=device)

X_tr, Y_tr = X_all[train_mask_t], Y_all[train_mask_t]
X_val, Y_val = X_all[val_mask_t], Y_all[val_mask_t]

print(f"TRAIN {len(X_tr):,} pts, VAL {len(X_val):,} pts (hold‑out hours ≥{hold_hr}/{H})")

# ╔════════ 2. NETWORK ═══════════════════════════════════════════════════════╗
class MLP(nn.Module):
    def __init__(self, hidden=64, depth=5):
        super().__init__()
        layers, inp = [], 5        # 5 inputs: x,y,t,u10,v10
        for _ in range(depth):
            layers += [nn.Linear(inp, hidden), nn.Tanh()]
            inp = hidden
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.D   = nn.Parameter(torch.tensor(0.01))  # learnable diffusivity

    def forward(self, x):              # safe – avoid inf/NaN in forward pass
        return torch.nan_to_num(self.net(x), nan=0.0, posinf=0.0, neginf=0.0)

pinn = MLP().to(device)

# ── optional warm start ──────────────────────────────────────────────────────
ckpt_path = pathlib.Path("results/ckpt_dataonly.pt")
if ckpt_path.exists():
    pinn.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("✓ warm‑started from", ckpt_path)
else:
    print("⚠️  no data‑only checkpoint found – starting from scratch")

optim = torch.optim.Adam(pinn.parameters(), lr=1e-4)

# ╔════════ 3. PDE RESIDUAL FUNCTION ═════════════════════════════════════════╗
def pde_residual(xc):
    """ ‖Ct + u·∇C − D∇²C‖² averaged on collocation points xc """
    xc = xc.clone().requires_grad_(True)     # (B,5)
    C  = pinn(xc)

    grad = torch.autograd.grad(C, xc, torch.ones_like(C),
                               create_graph=True, retain_graph=True)[0]
    dCdx, dCdy, dCdt = grad[:,0:1], grad[:,1:2], grad[:,2:3]

    d2Cdx2 = torch.autograd.grad(dCdx, xc, torch.ones_like(dCdx),
                                 create_graph=True, retain_graph=True)[0][:,0:1]
    d2Cdy2 = torch.autograd.grad(dCdy, xc, torch.ones_like(dCdy),
                                 create_graph=True, retain_graph=True)[0][:,1:2]

    u, v = xc[:,3:4], xc[:,4:5]      # wind components already scaled
    resid = dCdt + u*dCdx + v*dCdy - pinn.D*(d2Cdx2 + d2Cdy2)
    return (resid**2).mean()

# ╔════════ 4. TRAINING ══════════════════════════════════════════════════════╗
loss_hist = []
for epoch in range(501):
    pinn.train(); optim.zero_grad()

    pred_tr = pinn(X_tr)
    data_L  = ((pred_tr - Y_tr)**2).mean()

    # PDE collocation on a random 10 k subset of TRAIN points
    idx_col = torch.randperm(len(X_tr))[:10_000]
    pde_L   = pde_residual(X_tr[idx_col])

    lam = 10.0 * min(1.0, epoch/200)   # ramp λ_pde 0→10 over first 200 epochs
    loss = data_L + lam * pde_L

    loss.backward()
    torch.nn.utils.clip_grad_norm_(pinn.parameters(), 5.0)
    optim.step()

    if epoch % 50 == 0:
        pinn.eval()
        val_L = ((pinn(X_val) - Y_val)**2).mean().item()
        loss_hist.append((epoch, data_L.item(), pde_L.item(), val_L))
        print(f"E{epoch:04d}  data={data_L.item():.3e}  pde={pde_L.item():.3e}  "
              f"λ={lam:4.1f}  val={val_L:.3e}")

# ╔════════ 5. SAVE & PLOT ═══════════════════════════════════════════════════╗
pathlib.Path("results").mkdir(exist_ok=True)
torch.save(pinn.state_dict(), "results/ckpt_phys.pt")
print("✓ saved → results/ckpt_phys.pt")

if loss_hist:
    ep, dL, pL, vL = zip(*loss_hist)
    plt.semilogy(ep, dL, label="data")
    plt.semilogy(ep, pL, label="pde")
    plt.semilogy(ep, vL, label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.tight_layout()
    plt.show()
