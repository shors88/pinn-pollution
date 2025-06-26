#!/usr/bin/env python
"""
Interpolate ERA-5 hourly fields to the 1-km grid for one day.
Usage:  python scripts/regrid_era5.py 2022-07-01
Creates: prep/era5_grid_YYYY-MM-DD.nc
"""
import sys, os, pathlib, xarray as xr

# ── args & paths ─────────────────────────────────────────────────────────────
if len(sys.argv) != 2:
    sys.exit("Usage: python scripts/regrid_era5.py YYYY-MM-DD")

day  = sys.argv[1]                         # e.g. 2022-07-01
mfile = f"data/era5_{day[:4]}_{day[5:7]}.grib"
grid  = xr.open_dataset("prep/blank_grid.nc")

if not os.path.exists(mfile):
    sys.exit(f"❌  {mfile} not found. Run get_era5.py {day[:4]} first.")

# ── open & rename cfgrib short-names ─────────────────────────────────────────
ds_hr = xr.open_dataset(mfile, engine="cfgrib")

rename_map = {
    "u10": "10m_u_component_of_wind", "10u": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind", "10v": "10m_v_component_of_wind",
    "t2m": "2m_temperature",          "2t":  "2m_temperature",
}
ds_hr = ds_hr.rename({k: v for k, v in rename_map.items() if k in ds_hr})

# ── slice one day & select vars ──────────────────────────────────────────────
ds_day  = ds_hr.sel(time=day)
subset  = ds_day[["10m_u_component_of_wind",
                  "10m_v_component_of_wind",
                  "2m_temperature"]]

# ── horizontal re-grid ───────────────────────────────────────────────────────
ds_grid = subset.interp(longitude=grid.lon, latitude=grid.lat)

# ── save ─────────────────────────────────────────────────────────────────────
pathlib.Path("prep").mkdir(exist_ok=True)
out = f"prep/era5_grid_{day}.nc"
ds_grid.to_netcdf(out)
print("✓ ERA-5 re-gridded →", out, ds_grid.dims)

