#!/usr/bin/env python
"""
Build a lon/lat grid (≈ 1 km spacing) for the Denver box
and save as NetCDF: prep/blank_grid.nc
"""
import pathlib, numpy as np, xarray as xr

west, east, south, north = map(float,
    pathlib.Path("scripts/box_denver.txt").read_text().split())

# 1 km in degrees: 1 km ≈ 0.009° lat; adjust lon by cos(lat)
km_deg_lat = 1 / 111.0
km_deg_lon = km_deg_lat / np.cos(np.radians((south + north) / 2))

nx = int(np.ceil((east  - west) / km_deg_lon)) + 1
ny = int(np.ceil((north - south) / km_deg_lat)) + 1

x = np.linspace(west,  east,  nx)
y = np.linspace(south, north, ny)

grid = xr.Dataset(coords={"lon": ("lon", x), "lat": ("lat", y)})
pathlib.Path("prep").mkdir(exist_ok=True)
grid.to_netcdf("prep/blank_grid.nc")

print(f"✓ wrote prep/blank_grid.nc  ({nx} × {ny} points)")

