#!/usr/bin/env python
# Usage: python scripts/get_era5.py 2022
import sys, pathlib, calendar, cdsapi, os

YEAR = sys.argv[1]
lon_min, lon_max, lat_min, lat_max = map(float,
    pathlib.Path("scripts/box_denver.txt").read_text().split())

pathlib.Path("data").mkdir(exist_ok=True)
VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

c = cdsapi.Client()

for month in range(1, 13):
    target = f"data/era5_{YEAR}_{month:02d}.grib"
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": VARIABLES,
            "year": YEAR,
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, calendar.monthrange(int(YEAR), month)[1] + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [lat_max, lon_min, lat_min, lon_max],  # N, W, S, E
            "format": "grib",
        },
        target,
    )
    print("✓ downloaded", target)

