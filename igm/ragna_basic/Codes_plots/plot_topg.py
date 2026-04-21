import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIG ===


date_simu = "2026-01-29/17-12-45"


simu_path = os.path.join( "../outputs", date_simu)
out_dir = os.path.join(simu_path, "Plots")
os.makedirs(out_dir, exist_ok=True)
nc_path = os.path.join(simu_path, "output.nc")


Dropbox = False

# === LOAD DATA ===
ds = xr.open_dataset(nc_path)
thk = ds['topg']   # (time, y, x)
time = ds['time']

topg_last = thk.isel(time=-1).values

#moyenne = np.mean(slide_valid)
bed_min = np.percentile(topg_last, 10)
bed_max = np.percentile(topg_last, 90)

print( "Bed_min percantile 10 : ", bed_min, " and max ", bed_max)

# === Define a step ===

step = 1
selected_indices = range(0, len(time), step)

# === PLOT EACH YEAR ===
for i in selected_indices:
    year = int(time[i].values) if np.issubdtype(time[i].values.dtype, np.number) else str(time[i].values)
    print(f"year {year}")
    plt.figure(figsize=(8, 6))
    thk.isel(time=i).plot(cmap="terrain")   # "_r" makes thicker ice darker
    plt.title(f"Bed rock altitude - Year {year}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/bedrock_{year}.png", dpi=150)
    if year == 2021 :
        plt.show()
    
    plt.close()

if Dropbox : 
        with open("../../../../token.txt", "r") as f: 
            token = f.read().strip()
        dbx = dropbox.Dropbox(token)
        with open(f"{out_dir}/thickness_2020.png", "rb") as f:
            dbx.files_upload(f.read(), "/PhD_Lucie/IGM_Fig/plot.png", mode=dropbox.files.WriteMode.overwrite)

