#!/usr/bin/env python
import pathlib, numpy as np, pandas as pd, xarray as xr
from pykrige.ok import OrdinaryKriging

# ---- 1. Load grid & sensor snapshot -----------------------------------------
grid = xr.open_dataset("prep/blank_grid.nc")          # coords: lat, lon
df   = pd.read_parquet("raw/purpleair_latest.parquet").dropna(subset=["pm2.5_10minute"])

# ---- 2. Krige to the grid ----------------------------------------------------
OK = OrdinaryKriging(
    df.longitude, df.latitude, df["pm2.5_10minute"],
    variogram_model="linear", verbose=False, enable_plotting=False
)
# PyKrige returns   shape = (len(lat), len(lon))
z, _ = OK.execute("grid", grid.lon.values, grid.lat.values)

# ---- 3. Wrap in DataArray with explicit dims ---------------------------------
pm25 = xr.DataArray(
    z,
    coords={"lat": grid.lat, "lon": grid.lon},
    dims=("lat", "lon"),
    name="pm25"
)

# ---- 4. Save with timestamp --------------------------------------------------
ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H")
out = f"prep/pm25_grid_{ts}.nc"
pm25.to_dataset().to_netcdf(out)
print("✓ kriged PM₂.₅ →", out, pm25.shape)

