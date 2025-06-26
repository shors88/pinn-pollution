#!/usr/bin/env python
"""
Download the latest PurpleAir PM2.5 readings inside our metro box
and save them to raw/purpleair_latest.parquet
"""
import sys, pathlib, requests, pandas as pd

# ── 1. Bounding box (west, east, south, north) ────────────────────────────────
west, east, south, north = map(float,
    pathlib.Path("scripts/box_denver.txt").read_text().split())

# ── 2. PurpleAir API key (hard-coded) ─────────────────────────────────────────
APIKEY = "11134252-5218-11F0-81BE-42010A80001F"   # ← your Read/Enabled key

# ── 3. Compose and send the request ───────────────────────────────────────────
url = (
    "https://api.purpleair.com/v1/sensors"
    "?fields=latitude,longitude,pm2.5_10minute"
    f"&nwlng={west}&nwlat={north}&selng={east}&selat={south}"
    "&location_type=0"        # outdoor sensors only
)

resp = requests.get(url, headers={"X-API-Key": APIKEY})
payload = resp.json()

if "data" not in payload:
    print("❌ API returned an error:", payload)
    sys.exit(1)

data, cols = payload["data"], payload["fields"]
df = pd.DataFrame(data, columns=cols)

# ── 4. Save to Parquet ────────────────────────────────────────────────────────
pathlib.Path("raw").mkdir(exist_ok=True)
df.to_parquet("raw/purpleair_latest.parquet", engine="pyarrow")
print(f"✓ wrote {len(df):,} rows to raw/purpleair_latest.parquet")

