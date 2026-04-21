import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator, splprep, splev
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
# ----------------------------
# Paths and output directory
# ----------------------------
date_simu = "2026-04-15/11-36-42"



simu_path = os.path.join( "../outputs", date_simu)
out_dir = os.path.join("../outputs", date_simu, "Plots/velocities")
os.makedirs(out_dir, exist_ok=True)

which_flowline = "ragna"
flowline_file = os.path.join("../data/", f"flowline_ragna.csv")


point_fin = 8.2e3 # en m
smooth = True
# ----------------------------
# Load flowline from CSV
# ----------------------------
flowline_df = pd.read_csv(flowline_file)
x_flow = flowline_df['x'].values
y_flow = flowline_df['y'].values


# Smooth the line
if smooth :
    tck, _ = splprep(np.vstack([x_flow, y_flow]), s=0, k=min(3, len(x_flow)-1))
    n_samples = 200
    u_new = np.linspace(0, 1, n_samples)
    x_flow, y_flow = splev(u_new, tck)



# Compute cumulative distance along flowline
dist_flow = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x_flow)**2 + np.diff(y_flow)**2))])

# Find closest index to the target distance
idx_point_fin = np.argmin(np.abs(dist_flow - point_fin))
x_point_fin = x_flow[idx_point_fin]
y_point_fin = y_flow[idx_point_fin]
print(f"Point at {point_fin/1000:.2f} km from start:")
print(f"  x = {x_point_fin:.2f}, y = {y_point_fin:.2f}")

# ----------------------------
# Load model output
# ----------------------------
ds = xr.open_dataset(os.path.join(simu_path, "output.nc"))
u_var = ds['uvelsurf']
v_var = ds['vvelsurf']
x = ds['x'].values
y = ds['y'].values
time = ds['time']

# Convert time to years (robust)
years = np.array([int(t) for t in time.values])

ntime = len(years)
print(f"{ntime} time steps loaded, example years: {years[:5]} ... {years[-5:]}")

# ----------------------------
# Mask
# ----------------------------
nom_glacier = "Ragna-Mariebreen"


#path_shp_1990 = "~/PhD_Lucie/DATA/GLACIER_OUTLINES/CryoClim_GAO_SJ_1990/"
#shp_path_1990 = os.path.join(path_shp_1990, "CryoClim_GAO_SJ_1990.shp")
#gdf = gpd.read_file(shp_path_1990)

#gdf_one = gdf[gdf["NAME"] == nom_glacier]
#transform_sub = from_bounds(x_min, y_min, x_max, y_max, x, y)
#ny, nx = u_var.shape
#shapes = [(geom, 1) for geom in gdf_one.geometry]



#print(f"---mask_main shape (only {nom_glacier}):", .shape)
#print("   Nb pixels glacier:", np.sum(mask_main))
# ----------------------------
# Ensure y is monotonically increasing
# ----------------------------
reverse_y = False
if y[0] > y[-1]:
    y = y[::-1]
    reverse_y = True

# ----------------------------
# Interpolate velocity along flowline
# ----------------------------
speed_along_flow = np.full((ntime, len(x_flow)), np.nan)

for it in range(ntime):
    u_t = u_var.isel(time=it).values
    v_t = v_var.isel(time=it).values
    if reverse_y:
        u_t = u_t[::-1, :]
        v_t = v_t[::-1, :]
    speed_t = np.sqrt(u_t**2 + v_t**2)

    interp = RegularGridInterpolator((y, x), speed_t,
                                     bounds_error=False, fill_value=np.nan)
    pts = np.vstack([y_flow, x_flow]).T
    speed_along_flow[it, :] = interp(pts)

print("Velocity interpolation along the flowline completed. Shape:", speed_along_flow.shape)

# ----------------------------
# Plot velocity profiles along flowline (every other year)
# ----------------------------
years_to_plot = years[::2]
cmap = plt.colormaps.get('turbo_r')
norm = mcolors.Normalize(vmin=years_to_plot.min(), vmax=years_to_plot.max())

plt.figure(figsize=(10, 6))
for yr in years_to_plot:
    idxs = np.where(years == yr)[0]
    if len(idxs) == 0:
        continue
    idx = idxs[0]
    # Limiter la distance jusqu’au point_fin
    dist_mask = dist_flow <= point_fin

    plt.plot(dist_flow[dist_mask] / 1000.0,
             speed_along_flow[idx, dist_mask],
             color=cmap(norm(yr)),
             label=str(yr),
             linewidth=2)

    #plt.plot(np.linspace(0, 1, len(x_flow)) * np.sum(np.sqrt(np.diff(x_flow)**2 + np.diff(y_flow)**2)) / 1000,
     #        speed_along_flow[idx, :],
      #       color=cmap(norm(yr)),
       #      label=str(yr),
        #     linewidth=2)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=plt.gca(), label="Year")
plt.xlabel("Distance along flowline (km)")
plt.ylabel("Velocity (m/an)")
plt.title(f"Velocity profiles along the flowline {which_flowline}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"velocity_profiles_along_{which_flowline}_flowline.png"), dpi=200)
plt.show()

# ----------------------------
# Plot velocity field + normalized quiver + flowline
# ----------------------------
i_last = -1  # last time step
u_last = u_var.isel(time=i_last).values
v_last = v_var.isel(time=i_last).values
if reverse_y:
    u_last = u_last[::-1, :]
    v_last = v_last[::-1, :]
    y_plot = y
else:
    y_plot = y

speed_last = np.sqrt(u_last**2 + v_last**2)

# normalize vectors for homogeneous quiver
u_dir = u_last / np.where(speed_last == 0, np.nan, speed_last)
v_dir = v_last / np.where(speed_last == 0, np.nan, speed_last)

# optional: smooth flowline using spline
tck, _ = splprep(np.vstack([x_flow, y_flow]), s=0, k=min(3, len(x_flow)-1))
n_samples = 200
u_new = np.linspace(0, 1, n_samples)
x_flow_smooth, y_flow_smooth = splev(u_new, tck)


alpha = np.where(speed_last > 0, 1.0, 0.0)
plt.figure(figsize=(10, 8))
plt.pcolormesh(x, y_plot, speed_last, shading='auto', cmap='viridis', alpha=alpha, vmin=0, vmax=10)
plt.colorbar(label="Velocity (m/an)")

X, Y = np.meshgrid(x, y_plot)
step = 10

Q = plt.quiver(X[::step, ::step], Y[::step, ::step],
               u_dir[::step, ::step], v_dir[::step, ::step],
               scale=40, color='k', width=0.002)

# Use Q in the quiverkey call
plt.quiverkey(Q, 0.88, 1.02, 1,
              label="Flow direction (norm.)",
              labelpos='E')

plt.plot(x_flow_smooth, y_flow_smooth, 'r-', linewidth=2.5, label="Flowline")
#plt.scatter(x_flow[::20], y_flow[::20], c='k', s=4, zorder=5, label="Flowline points")
plt.scatter(x_point_fin, y_point_fin, color='green', edgecolor='black', s=20, zorder=5, label=f"Point @ {point_fin/1000:.1f} km")

plt.title(f"Velocity field and {which_flowline} flowline – Year {years[i_last]}")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"velocity_field_flowline_{which_flowline}_{years[i_last]}.png"), dpi=200)
plt.show()



plt.figure(figsize=(10, 8))

# ---- Background velocity amplitude ----
plt.pcolormesh(x, y_plot, speed_last, shading='auto',
               cmap='rainbow', alpha=alpha, vmin=0, vmax=10)
plt.colorbar(label="Velocity (m/an)")

# ---- Grid ----
X, Y = np.meshgrid(x, y_plot)

# -------------------------------
# STREAMPLOT
# -------------------------------
plt.streamplot(
    X, Y,
    u_dir, v_dir,            # direction field
    density=2,             # increase for more curves
    color='k',               # streamline color
    linewidth=0.8,           # line thickness
    arrowsize=0.8            # small arrowheads on the streamlines
)

# ---- Plot flowline ----
#plt.plot(x_flow_smooth, y_flow_smooth,         'r-', linewidth=2.5, label="Flowline")

# ---- Point_fin ----
#plt.scatter(x_point_fin, y_point_fin,            color='green', edgecolor='black', s=20, zorder=5,label=f"Point @ {point_fin/1000:.1f} km")

plt.title(f"Velocity field and streamlines – Year {years[i_last]}")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig(os.path.join(out_dir,
            f"velocity_field_flowline_{which_flowline}_{years[i_last]}_streamlines.png"),
            dpi=200)
plt.show()


# ----------------------------
# Save flowline for reuse
# ----------------------------
#flowline_saved = pd.DataFrame({'x': x_flow, 'y': y_flow})
#flowline_saved.to_csv(os.path.join(out_dir, f"flowline_{which_flowline}_2022_saved.csv"), index=False)
#print("Flowline saved to CSV for reuse.")
