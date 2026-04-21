import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIG ===
nc_path = "../data/input.nc"   # path to your NetCDF file
out_dir = "../Plots"
os.makedirs(out_dir, exist_ok=True)

# === LOAD DATA ===
ds = xr.open_dataset(nc_path)
usurf = ds['usurf']  
topg = ds['topg']  
thk = usurf - topg  # (time, y, x)


# === PLOT EACH YEAR ===
     
plt.figure(figsize=(8, 6))
thk.plot(cmap="Blues_r")   # "_r" makes thicker ice darker
plt.title(f"Ice Thickness Input")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig(f"{out_dir}/thickness_input.png", dpi=150)
plt.show()
plt.close()
