import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator, splprep, splev
import rasterio
from rasterio.transform import from_origin

# ----------------------------
# Paths and output directory
# ----------------------------
date_simu = "2026-01-28/16-00-18/"



simu_path = os.path.join("../outputs", date_simu)
out_dir = os.path.join("../outputs", date_simu, "Plots/geometry")
os.makedirs(out_dir, exist_ok=True)

which_flowline = "werenskiold_south"
flowline_file = os.path.join("../data", f"flowline_{which_flowline}_2022.csv")
smooth = True

enthalpy = "without"

# ----------------------------
# Load flowline
# ----------------------------
flowline_df = pd.read_csv(flowline_file)
x_flow = flowline_df["x"].values #- 250000 -1718
y_flow = flowline_df["y"].values #-2390 
print("--- FLOWLINE ---")
print("xmin :", np.min(x_flow), ", xmax :", np.max(x_flow))


# Smooth the flowline if needed
if smooth:
    tck, _ = splprep(np.vstack([x_flow, y_flow]), s=0, k=min(3, len(x_flow)-1))
    u_new = np.linspace(0, 1, 200)
    x_flow, y_flow = splev(u_new, tck)

# Distance along flowline
dist_flow = np.concatenate(
    [[0], np.cumsum(np.sqrt(np.diff(x_flow) ** 2 + np.diff(y_flow) ** 2))]
)

# ----------------------------
# Load model output
# ----------------------------
ds = xr.open_dataset(os.path.join(simu_path, "output.nc"))
print(ds)


thk = ds["thk"]
topg = ds["topg"]
usurf = ds["usurf"]

x = ds["x"].values
y = ds["y"].values
time = ds["time"]

print("--- NETCDF ---")
print("xmin", np.min(x), ", xmax :", np.max(x))

# Convert time to years robustly
years = np.array([int(t) for t in time.values])
ntime = len(years)

print(f"{ntime} time steps loaded, years from {years.min()} to {years.max()}")

# ----------------------------
# Ensure y is increasing
# ----------------------------
reverse_y = False
if y[0] > y[-1]:
    y = y[::-1]
    reverse_y = True

# ----------------------------
# Interpolate speed along flowline
# ----------------------------
thk_along_flow = np.full((ntime, len(x_flow)), np.nan)

for it in range(ntime):
    h_t = thk.isel(time=it).values
    
    if reverse_y:
        h_t = u_t[::-1, :]
        
    interp = RegularGridInterpolator(
        (y, x), h_t,
        bounds_error=False, fill_value=np.nan
    )

    pts = np.vstack([y_flow, x_flow]).T
    thk_along_flow[it, :] = interp(pts)

print("Velocity interpolation completed. Shape:", thk_along_flow.shape)

# ----------------------------
# Plot velocity profiles along flowline
# ----------------------------
years_to_plot = years[::2]
cmap = plt.colormaps.get("turbo_r")
norm = mcolors.Normalize(vmin=years_to_plot.min(), vmax=years_to_plot.max())

plt.figure(figsize=(10, 6))

for yr in years_to_plot:
    idx = np.where(years == yr)[0]
    if len(idx) == 0:
        continue
    idx = idx[0]

    plt.plot(dist_flow / 1000.0,
             thk_along_flow[idx, :],
             color=cmap(norm(yr)),
             linewidth=2,
             label=str(yr))

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
ax = plt.gca()
plt.colorbar(sm, ax=ax, label="Year")

plt.xlabel("Distance along flowline (km)")
plt.ylabel("Thickness (m)")
plt.title(f"Thickness profile along the flowline for Storglaciaren, {enthalpy} enthalpy")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"thickness_profiles_{enthalpy}_enthalpy.png"), dpi=200)
plt.show()

# ----------------------------
# Velocity field + quiver + flowline
# ----------------------------

i_last = -1
h_last = thk.isel(time=i_last).values

if reverse_y:
    h_last = h_last[::-1, :]
    y_plot = y
else:
    y_plot = y



# Smooth flowline for plot
tck, _ = splprep(np.vstack([x_flow, y_flow]), s=0, k=min(3, len(x_flow)-1))
x_flow_s, y_flow_s = splev(np.linspace(0, 1, 200), tck)


alpha = np.where(h_last == 0, 0.0, 1.0)

plt.figure(figsize=(10, 8))
plt.pcolormesh(x, y_plot, h_last,
               shading="auto", cmap="Blues",
               vmin=0, vmax=np.nanmax(h_last), alpha=alpha)

plt.colorbar(label="Thickness (m)")


plt.plot(x_flow_s, y_flow_s, "r-", linewidth=2.5, label="Flowline")

plt.xlim(504600, 513900)   
plt.ylim(8.5522e6, 8.5599e6)

plt.title(f"Thickness and flowline, {enthalpy} enthalpy — Year {years[i_last]}")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"thickness_field_{years[i_last]}.png"), dpi=200)
plt.show()


# ----------------------------
# Bedrock
# ----------------------------



zb_last = topg.isel(time=i_last).values

if reverse_y:
    zb_last = zb_last[::-1, :]
    y_plot = y
else:
    y_plot = y



# Smooth flowline for plot


#alpha = np.where(h_last == 0, 0.0, 1.0)

plt.figure(figsize=(10, 8))
plt.pcolormesh(x, y_plot, zb_last,
               shading="auto", cmap="terrain",
               vmin=0, vmax=np.nanmax(zb_last), alpha=alpha)

plt.colorbar(label="Bedrock altitude (m)")


plt.plot(x_flow_s, y_flow_s, "r-", linewidth=2.5, label="Flowline")

plt.xlim(396000, 400000)
plt.ylim(7.533e6, 7.536e6)

plt.title(f"Bedrock attitude and flowline, {enthalpy} enthalpy — Year {years[i_last]}")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"bedrock_{years[i_last]}.png"), dpi=200)
plt.show()

