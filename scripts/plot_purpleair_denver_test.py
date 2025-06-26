import pandas as pd, matplotlib.pyplot as plt
df = pd.read_parquet('raw/purpleair_latest.parquet')
plt.scatter(df.longitude, df.latitude, c=df['pm2.5_10minute'])
plt.colorbar(); plt.show()
