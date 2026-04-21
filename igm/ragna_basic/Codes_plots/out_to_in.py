import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIG ===
nc_path = "../outputs/2025-10-13/11-18-27/output.nc"   # path to your NetCDF file
nc_path_icemask = "../data/input.nc" 
out_dir = "../New_input"
os.makedirs(out_dir, exist_ok=True)

# === LOAD DATA ===
ds = xr.open_dataset(nc_path)
ds_icemask = xr.open_dataset(nc_path_icemask)

x = ds['x']
y = ds['y']

#thk = ds['thk']   # (time, y, x)
#usurf = ds['usurf']
icemask = ds_icemask['icemaskobs'].data
time = ds['time']


# === Define a year ===

selected_year = len(time)-1
thk = ds['thk'].isel(time=selected_year).data   # (time, y, x)
usurf = ds['usurf'].isel(time=selected_year).data



# === Create a new dataset ===

ds_new = xr.Dataset(
        {"usurf": (("y", "x"), usurf),
         "thk": (("y", "x"), thk),
         "icemask": (("y", "x"), icemask),

            },
        coords={
            "x":x,
            "y":y}

        )
print("New dataset saved")
output_file = os.path.join(out_dir, "input_new.nc")
ds_new.to_netcdf(output_file)

# === PLOT EACH YEAR ===
i = selected_year
plt.figure(figsize=(8, 6))
thk.plot(cmap="Blues_r")   # "_r" makes thicker ice darker
plt.title(f"Ice Thickness - Year {year}")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig(f"{out_dir}/thickness_input.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
usurf.isel(time=i).plot(cmap="cool")   
plt.title(f"Surface topography - Year {year}")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig(f"{out_dir}/thickness_{year}.png", dpi=150)
plt.close()


