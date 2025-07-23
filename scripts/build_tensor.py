#!/usr/bin/env python
"""
build_tensor.py   (1–7 July 2023 Denver)

Creates a flat tensor for the PINN:

    X  →  [lon_norm, lat_norm, t_norm, u10_z, v10_z]
    Y  →  pm25
    t_idx → integer hour index (0 … 167)

Inputs
------
raw/epa/by_hour/hourly_2023070[1‑7]_HH.csv
prep/era5_grid_2023-07-0[1‑7].nc

Output
------
prep/train_tensor.npz
"""

import glob, pathlib
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# ─── 1 ▸ locate per‑hour CSVs ───────────────────────────────────────────────
epa_files = sorted(glob.glob("raw/epa/by_hour/hourly_2023070[1-7]_*.csv"))
if not epa_files:
    raise SystemExit("❌  No EPA hourly CSVs found for 2023‑07‑01…07")

era5_daily = {f"2023-07-{d:02d}": f"prep/era5_grid_2023-07-{d:02d}.nc"
              for d in range(1, 8)}

X_parts, Y_parts, t_parts = [], [], []

# ─── 2 ▸ loop over hours ────────────────────────────────────────────────────
for ti, f in enumerate(epa_files):
    df = pd.read_csv(
        f,
        usecols=["latitude", "longitude", "sample_measurement"],
        dtype={"latitude": "float32",
               "longitude": "float32",
               "sample_measurement": "float32"},
    ).rename(columns={"sample_measurement": "pm25"})
    if df.empty:
        continue

    # day & hour from file name
    stem = Path(f).stem                    # e.g. hourly_20230701_00
    day  = stem[7:15]                      # 20230701
    hour = int(stem[-2:])                  # 00 … 23
    nc   = era5_daily[f"{day[:4]}-{day[4:6]}-{day[6:]}"]

    winds = xr.open_dataset(nc).isel(time=hour)
    # spatial mean winds over domain
    u10 = float(winds["10m_u_component_of_wind"].mean())
    v10 = float(winds["10m_v_component_of_wind"].mean())

    n = len(df)
    X_parts.append(np.column_stack([
        df.longitude.values,
        df.latitude .values,
        np.full(n, ti, dtype=np.float32),      # raw hour index
        np.full(n, u10, dtype=np.float32),
        np.full(n, v10, dtype=np.float32),
    ]))
    Y_parts.append(df.pm25.values[:, None])
    t_parts.append(np.full(n, ti, dtype=np.int32))

# ─── 3 ▸ concat & normalise ─────────────────────────────────────────────────
X_all = np.vstack(X_parts).astype(np.float32)
Y_all = np.vstack(Y_parts).astype(np.float32)
t_idx = np.concatenate(t_parts)

# scale lon, lat, t to [‑1, 1]
for col in (0, 1, 2):
    mn, rng = X_all[:, col].min(), np.ptp(X_all[:, col])
    X_all[:, col] = 2 * (X_all[:, col] - mn) / rng - 1

# z‑score winds
for col in (3, 4):
    mu, sig = X_all[:, col].mean(), X_all[:, col].std()
    X_all[:, col] = (X_all[:, col] - mu) / sig

# ─── 4 ▸ save tensor ────────────────────────────────────────────────────────
pathlib.Path("prep").mkdir(exist_ok=True)
np.savez_compressed(
    "prep/train_tensor.npz",
    X=X_all,
    Y=Y_all,
    t_idx=t_idx
)
print("✓ Wrote prep/train_tensor.npz:",
      f"samples={X_all.shape[0]} dims={X_all.shape[1]} hours={len(epa_files)}")

