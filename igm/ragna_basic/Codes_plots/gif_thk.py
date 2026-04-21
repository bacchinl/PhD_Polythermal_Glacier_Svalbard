import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os

# === CONFIG ===
nc_path = "..//outputs/2025-10-21/12-07-09/output.nc"   # Path to your NetCDF
out_dir = "Plots/GIF_thk"
gif_name = "thickness_werenskiold_evolution.gif"
step = 1  # Plot every 10th year
fps = 3    # Frames per second for the GIF

os.makedirs(out_dir, exist_ok=True)

# === LOAD DATA ===
ds = xr.open_dataset(nc_path)
thk = ds['thk']   # (time, y, x)
time = ds['time']

# Determine time indices
indices = range(0, len(time), step)

# Fix colorbar scale across all frames
vmin = float(thk.min())
vmax = float(thk.max())

frames = []

# === LOOP OVER YEARS ===
for i in indices:
    year = time[i].values
    if np.issubdtype(time.dtype, np.datetime64):
        year_label = np.datetime_as_string(year, unit='Y')
    else:
        year_label = str(int(year))

    plt.figure(figsize=(8, 6))
    thk_slice = thk.isel(time=i)
    thk_slice.plot(cmap="Blues_r", vmin=vmin, vmax=vmax, add_colorbar=True)
    plt.title(f"Ice Thickness - Year {year_label}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    # Save temporary frame
    frame_path = os.path.join(out_dir, f"thk_{i:03d}.png")
    plt.savefig(frame_path, dpi=150)
    plt.close()

    frames.append(imageio.imread(frame_path))

# === CREATE GIF ===
imageio.mimsave(gif_name, frames, fps=fps)
print(f" Animation saved as {gif_name} ({len(frames)} frames, step={step} years)")
