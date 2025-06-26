#!/usr/bin/env python
"""
Minimal 2-D advection–diffusion PINN for Denver PM2.5.

Inputs : (x, y, t)  scaled to [-1, 1]
Output : Ĉ(x, y, t)  (PM₂.₅ concentration)
Learned : effective diffusivity D
Loss    : data-fit  +  PDE residual  +  zero-flux boundary
"""

import torch, torch.nn as nn
import numpy as np, pandas as pd, xarray as xr, matplotlib.pyplot as plt
import pathlib, random, os, warnings

warnings.filterwarnings("ignore", category=FutureWarning)   # xarray timedelta msg

# reproducibility ─────────────────────────────────────────────────────────────
torch.manual_seed(0);  np.random.seed(0);  random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
tt = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)

# ╔════════ 1. LOAD AND PREP DATA ═════════════════════════════════════════════╗
grid = xr.open_dataset("prep/blank_grid.nc")

# PM2.5 snapshot (kriged)  ── fill any NaNs just in case
pm25 = xr.open_dataset(sorted(pathlib.Path("prep").glob("pm25_grid_*.nc"))[0])
pm25 = pm25.fillna(pm25.pm25.mean())                        # safety fill

# ERA-5 weather, first hour
era5 = xr.open_dataset("prep/era5_grid_2022-07-01.nc")
u10  = era5["10m_u_component_of_wind"][0].values
v10  = era5["10m_v_component_of_wind"][0].values

# flatten grid to (N,2)
Xg, Yg = np.meshgrid(grid.lon, grid.lat)
coords = np.stack([Xg.ravel(), Yg.ravel()], 1)              # (N,2)

# scale x,y to [-1,1]  (use np.ptp to support NumPy ≥2)
xn = (coords[:,0] - coords[:,0].mean()) / np.ptp(coords[:,0])
yn = (coords[:,1] - coords[:,1].mean()) / np.ptp(coords[:,1])
tn = np.zeros_like(xn)                                      # single snapshot ⇒ t=0
Cn = pm25["pm25"].values.ravel()

# mask possible NaNs (should be none after fill, but safe)
mask = ~np.isnan(Cn)
xn, yn, tn, Cn = xn[mask], yn[mask], tn[mask], Cn[mask]

# train/val split
N = len(xn)
idx = np.random.permutation(N)
train, val = idx[: int(.9*N)], idx[int(.9*N):]

x_train = tt(np.stack([xn[train], yn[train], tn[train]], 1))
y_train = tt(Cn[train][:,None])
x_val   = tt(np.stack([xn[val],   yn[val],   tn[val]],   1))
y_val   = tt(Cn[val][:,None])

# ╔════════ 2. NETWORK ════════════════════════════════════════════════════════╗
class MLP(nn.Module):
    def __init__(self, hidden=64, depth=5):
        super().__init__()
        layers, inp = [], 3
        for _ in range(depth):
            layers += [nn.Linear(inp, hidden), nn.Tanh()]
            inp = hidden
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.D   = nn.Parameter(torch.tensor(0.01))        # learnable diff.

    def forward(self, x): return self.net(x)

pinn  = MLP().to(device)
optim = torch.optim.Adam(pinn.parameters(), lr=1e-3)

# ╔════════ 3. HELPER LOSSES ══════════════════════════════════════════════════╗
def residual_loss(xc):
    """‖Ct + u·∇C − D∇²C‖² on random collocation points."""
    xc = xc.clone().requires_grad_(True)
    C  = pinn(xc)

    grad = torch.autograd.grad(C, xc, torch.ones_like(C),
                               create_graph=True, retain_graph=True)[0]
    dCdx, dCdy, dCdt = grad[:,0:1], grad[:,1:2], grad[:,2:3]

    d2Cdx2 = torch.autograd.grad(dCdx, xc, torch.ones_like(dCdx),
                                 create_graph=True, retain_graph=True)[0][:,0:1]
    d2Cdy2 = torch.autograd.grad(dCdy, xc, torch.ones_like(dCdy),
                                 create_graph=True, retain_graph=True)[0][:,1:2]

    # winds: use nan-robust mean
    u = torch.nanmean(tt(u10))
    v = torch.nanmean(tt(v10))
    if torch.isnan(u): u = torch.tensor(0., device=device)
    if torch.isnan(v): v = torch.tensor(0., device=device)

    resid = dCdt + u*dCdx + v*dCdy - pinn.D*(d2Cdx2+d2Cdy2)
    return torch.mean(resid**2)

def boundary_loss():
    """Zero-flux (∂C/∂y) on N/S edges."""
    top = tt(np.stack([xn.reshape(Xg.shape)[0,:],
                       yn.reshape(Yg.shape)[0,:],
                       np.zeros_like(Xg[0,:])],1))
    bot = tt(np.stack([xn.reshape(Xg.shape)[-1,:],
                       yn.reshape(Yg.shape)[-1,:],
                       np.zeros_like(Xg[-1,:])],1))
    pts  = torch.cat([top, bot], 0).requires_grad_(True)
    C    = pinn(pts)
    grad = torch.autograd.grad(C, pts, torch.ones_like(C),
                               create_graph=True)[0]
    flux = grad[:,1:2] * torch.sign(pts[:,1:2])             # only y-component
    return torch.mean(flux**2)

# ╔════════ 4. TRAINING LOOP ══════════════════════════════════════════════════╗
loss_hist = []
for epoch in range(1000):
    pinn.train();  optim.zero_grad()

    data_L = torch.mean((pinn(x_train) - y_train)**2)
    pde_L  = residual_loss(tt(np.random.uniform(-1,1,(5000,3))))
    bnd_L  = boundary_loss()
    loss   = data_L + 1.0*pde_L + 0.1*bnd_L

    # guard against NaNs
    if torch.isnan(loss):
        print(f"NaN @epoch {epoch} — skipping step");  continue

    loss.backward();  optim.step()

    # log every 50 epochs
    if epoch % 50 == 0:
        pinn.eval()
        val_L = torch.mean((pinn(x_val)-y_val)**2).item()
        loss_hist.append((epoch, loss.item(), val_L))
        print(f"E{epoch:04d}  total={loss.item():.4f}  "
              f"data={data_L.item():.4f}  val={val_L:.4f}")

# ╔════════ 5. PLOT LOSS CURVES ═══════════════════════════════════════════════╗
if loss_hist:
    ep, tr, va = zip(*loss_hist)
    plt.semilogy(ep, tr, label="train")
    plt.semilogy(ep, va, label="val")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.tight_layout()
    plt.show()
else:
    print("No finite losses recorded — check data for NaNs.")

