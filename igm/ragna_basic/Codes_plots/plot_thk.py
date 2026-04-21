import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os


# === CONFIG ===
nc_path = "../outputs/2026-04-20/12-11-45/output.nc"   # path to your NetCDF file
out_dir = "../Plots/Reconstruction_Weren"
os.makedirs(out_dir, exist_ok=True)

Dropbox = False

# === LOAD DATA ===
ds = xr.open_dataset(nc_path)
thk = ds['thk']   # (time, y, x)
time = ds['time']


# === Define a step ===

step = 1
selected_indices = range(0, len(time), step)

# === PLOT EACH YEAR ===
for i in selected_indices:
    year = int(time[i].values) if np.issubdtype(time[i].values.dtype, np.number) else str(time[i].values)
    plt.figure(figsize=(8, 6))
    thk.isel(time=i).plot(cmap="Blues_r")   # "_r" makes thicker ice darker
    plt.title(f"Ice Thickness - Year {year}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/thickness_{year}.png", dpi=150)
    if year == 2021 :
        plt.show()
    
    plt.close()

if Dropbox : 
        with open("../../../../token.txt", "r") as f: 
            token = f.read().strip()
        dbx = dropbox.Dropbox(token)
        with open(f"{out_dir}/thickness_2020.png", "rb") as f:
            dbx.files_upload(f.read(), "/PhD_Lucie/IGM_Fig/plot.png", mode=dropbox.files.WriteMode.overwrite)

