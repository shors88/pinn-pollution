#!/usr/bin/env python
"""
Merge kriged PM₂.₅ and ERA5 fields into a single (time, lat, lon, channel) array
and save npz ready for a PyTorch dataloader.
"""
import glob, numpy as np, xarray as xr, pathlib
from sklearn.preprocessing import StandardScaler

grid = xr.open_dataset("prep/blank_grid.nc")

# concat all daily files you generated in steps 6–7
pm25_files = sorted(glob.glob("prep/pm25_grid_*.nc"))
era5_files = [f"prep/era5_grid_{f.split('_')[-1]}" for f in pm25_files]

pm25 = xr.open_mfdataset(pm25_files, combine="nested", concat_dim="time")
era5 = xr.open_mfdataset(era5_files, combine="nested", concat_dim="time")

ds = xr.merge([pm25, era5])      # same coords; merges along time

# stack lat/lon so we can scale features easily
stacked = ds.stack(points=("lat", "lon"))

# scale independent variables
X_raw = stacked[["10m_u_component_of_wind",
                 "10m_v_component_of_wind",
                 "2m_temperature"]].to_array().transpose("time","points","variable")
Y_raw = stacked["pm25"].values[..., None]   # make last dim 1

scaler = StandardScaler().fit(X_raw.reshape(-1, X_raw.shape[-1]))
X_scaled = scaler.transform(X_raw.reshape(-1, X_raw.shape[-1])
                            ).reshape(X_raw.shape)

# save
np.savez_compressed(
    "prep/train_tensor.npz",
    X=X_scaled.astype(np.float32),  # (time, pts, 3)
    Y=Y_raw.astype(np.float32),     # (time, pts, 1)
    lon=grid.lon.values,
    lat=grid.lat.values
)
print("✓ training tensor → prep/train_tensor.npz  ",
      X_scaled.shape, Y_raw.shape)

