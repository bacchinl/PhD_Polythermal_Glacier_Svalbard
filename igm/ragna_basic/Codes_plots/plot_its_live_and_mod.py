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
import rioxarray
from affine import Affine
import contextily as ctx

# ----------------------------
# Paths and output directory
# ----------------------------
date_simu = "2026-04-27/12-43-53" # Poster : "2026-04-20/15-27-52"without "2026-04-20/12-11-45"with  




simu_path = os.path.join( "../outputs", date_simu)
out_dir = os.path.join("../outputs", date_simu, "Plots/velocities")
os.makedirs(out_dir, exist_ok=True)

which_flowline = "ragna"
flowline_file = os.path.join("../data/", f"centerline_ragna_EM_mod.csv")


point_fin = 7.2e3 # en m
smooth = True


# ----------------------------
# Extent
# ----------------------------
extent = {
    "xmin": 553000,
    "xmax": 558700,
    "ymin": 8633000,
    "ymax": 8640500
}

# ----------------------------
# Plot parameters
# ----------------------------
vmin = 0
vmax = 10  # à adapter
cmap = "viridis"

stream_density = 2


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
# Load obs
# ----------------------------
obs_dir = "~/PhD_Lucie/DATA/Velocities/Its_LIVE/ITS_LIVE_velocity_120m_RGI07A_0000_V02.1.nc"
obs_ds = xr.open_dataset(obs_dir)
obs_ds = obs_ds[["vx", "vy"]]

obs_ds = obs_ds.rio.write_crs("EPSG:3413")
obs_utm = obs_ds.rio.reproject("EPSG:32633")

u_obs = obs_utm["vx"].values
v_obs = obs_utm["vy"].values


x_obs = obs_utm["x"].values
y_obs = obs_utm["y"].values




speed_obs = np.sqrt(u_obs**2 + v_obs**2)

print("VX stats:")
print("mean =", obs_ds["vx"].mean().values)
print("min  =", obs_ds["vx"].min().values)
print("max  =", obs_ds["vx"].max().values)

print("\nVY stats:")
print("mean =", obs_ds["vy"].mean().values)
print("min  =", obs_ds["vy"].min().values)
print("max  =", obs_ds["vy"].max().values)

print("\n--- AFTER REPROJECTION ---")

print("VX mean =", obs_utm["vx"].mean().values)
print("VX min  =", obs_utm["vx"].min().values)
print("VX max  =", obs_utm["vx"].max().values)

print("\nVY mean =", obs_utm["vy"].mean().values)
print("VY min  =", obs_utm["vy"].min().values)
print("VY max  =", obs_utm["vy"].max().values)

print("\nMagn mean =", speed_obs.mean())
print("VY min  =", speed_obs.min())
print("VY max  =", speed_obs.max())


print("\nCRS =", obs_utm.rio.crs)





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

# Output_resolution
dx = x[1] - x[0]
dy = y[1] - y[0]

transform = Affine.translation(x.min(), y.min()) * Affine.scale(dx, dy)

# Dimensions (y, x)
ny = len(y)
nx = len(x)

# ----------------------------
# Mask
# ----------------------------
nom_glacier = "Ragna-Mariebreen"


path_shp_2010 = "~/PhD_Lucie/DATA/GLACIER_OUTLINES/CryoClim_GAO_SJ_2001-2010/"
shp_path_2010 = os.path.join(path_shp_2010, "CryoClim_GAO_SJ_2001-2010.shp")
gdf = gpd.read_file(shp_path_2010)

gdf_one = gdf[gdf["NAME"] == nom_glacier]

print("GDF ", nom_glacier,  gdf_one)

shapes_list = [(geom, 1) for geom in gdf_one.geometry]

mask = rasterize(
    shapes_list,
    out_shape=(ny, nx),
    transform=transform,
    fill=0,
    dtype='uint8'
)


print(f"---mask_main shape (only {nom_glacier}):", mask.shape)
print("   Nb pixels glacier:", np.sum(mask))
# ----------------------------
# Ensure y is monotonically increasing
# ----------------------------
reverse_y = False
if y[0] > y[-1]:
    y = y[::-1]
    reverse_y = True

# ----------------------------
# Prepare model data
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
speed_last = np.where(mask, speed_last, np.nan)

# normalize vectors for homogeneous quiver
u_dir = u_last / np.where(speed_last == 0, np.nan, speed_last)
v_dir = v_last / np.where(speed_last == 0, np.nan, speed_last)

ix = np.where((x >= extent["xmin"]) & (x <= extent["xmax"]))[0]
iy = np.where((y_plot >= extent["ymin"]) & (y_plot <= extent["ymax"]))[0]

x_sub = x[ix]
y_sub = y_plot[iy]

speed_sub = speed_last[np.ix_(iy, ix)]
u_dir_sub = u_dir[np.ix_(iy, ix)]
v_dir_sub = v_dir[np.ix_(iy, ix)]

# ----------------------------
# Prepare obs data
# ----------------------------

# interp obs on model grid
interp_u = RegularGridInterpolator(
    (y_obs, x_obs),
    u_obs,
    bounds_error=False,
    fill_value=np.nan
)

interp_v = RegularGridInterpolator(
    (y_obs, x_obs),
    v_obs,
    bounds_error=False,
    fill_value=np.nan
)

X, Y = np.meshgrid(x, y_plot)
points = np.stack([Y.ravel(), X.ravel()], axis=-1)

u_obs_on_model = interp_u(points).reshape(len(y_plot), len(x))
v_obs_on_model = interp_v(points).reshape(len(y_plot), len(x))

speed_obs_on_model = np.sqrt(u_obs_on_model**2 + v_obs_on_model**2)
speed_obs_on_model = np.where(mask, speed_obs_on_model, np.nan)

# extent on obs
speed_obs_sub = speed_obs_on_model[np.ix_(iy, ix)]
u_obs_sub = u_obs_on_model[np.ix_(iy, ix)]
v_obs_sub = v_obs_on_model[np.ix_(iy, ix)]

u_dir_obs = u_obs_sub / np.where(speed_obs_sub == 0, np.nan, speed_obs_sub)
v_dir_obs = v_obs_sub / np.where(speed_obs_sub == 0, np.nan, speed_obs_sub)

speed_diff= speed_sub -speed_obs_sub

fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# ==========================
# 1. MODELE
# ==========================
im0 = axes[0].imshow(
    speed_sub,
    extent=[x_sub.min(), x_sub.max(), y_sub.min(), y_sub.max()],
    origin="lower",
    cmap=cmap,
    vmin=vmin,
    vmax=vmax
)

axes[0].streamplot(
    x_sub,
    y_sub,
    u_dir_sub,
    v_dir_sub,
    color="k",
    density=stream_density,
    linewidth=0.5
)

axes[0].set_title("Modèle")

# ==========================
# 2. OBS
# ==========================
im1 = axes[1].imshow(
    speed_obs_sub,
    extent=[x_sub.min(), x_sub.max(), y_sub.min(), y_sub.max()],
    origin="lower",
    cmap=cmap,
    vmin=vmin,
    vmax=vmax
)

axes[1].streamplot(
    x_sub,
    y_sub,
    -u_dir_obs,
    v_dir_obs,
    color="k",
    density=stream_density,
    linewidth=0.5
)

axes[1].set_title("Observations")

# ==========================
# 3. DIFFERENCE
# ==========================
im2 = axes[2].imshow(speed_diff,extent=[x_sub.min(), x_sub.max(), y_sub.min(), y_sub.max()],origin="lower",cmap="RdBu_r", vmin = -10, vmax = 10)

axes[2].set_title("Mod - Obs")

# ==========================
# Colorbars
# ==========================
cbar = fig.colorbar(im0, ax=axes[:2], orientation="vertical", fraction=0.03)
cbar.set_label("Vitesse (m/an)")


cbar2 = fig.colorbar(im2, ax=axes[2], orientation="vertical", fraction=0.03)
cbar2.set_label("Différence (m/an)")

plt.show()















########## PLOT STREAMLINES #########
fig,axes = plt.subplots(1,3,figsize=(18, 6))


#### <PLOT MODEL
alpha_sub = np.where(speed_sub > 0, 1.0, 0.0)
# ---- Background velocity amplitude ----
im0=axes[0].pcolormesh(x_sub, y_sub, speed_sub, shading='auto',
               cmap='rainbow',alpha=alpha_sub, vmin=vmin, vmax=vmax)
cbar0=plt.colorbar(im0, ax=axes[0])

# ---- Grid ----
X, Y = np.meshgrid(x_sub, y_sub)

# -------------------------------
# STREAMPLOT
# -------------------------------
axes[0].streamplot(
    X, Y,
    u_dir_sub, v_dir_sub,            # direction field
    density=2,             # increase for more curves
    color='k',               # streamline color
    linewidth=0.8,           # line thickness
    arrowsize=0.8            # small arrowheads on the streamlines
)


ctx.add_basemap(
    axes[0],
    crs="EPSG:32633",
    source=ctx.providers.Esri.WorldImagery
)

axes[0].set_xlim(extent["xmin"], extent["xmax"])
axes[0].set_ylim(extent["ymin"], extent["ymax"])


axes[0].set_title(f"Velocity field and streamlines – Year 2024")
axes[0].set_xlabel("Longitude", fontsize = 14)
axes[0].set_ylabel("Latitude", fontsize = 14)
#ax.legend()

cbar0.set_ticks([0, 1,2,3])
cbar0.set_label("Velocity (m/a)", fontsize=16)
#cbar0.axes[0].tick_params(labelsize=12)



ax.set_autoscale_on(False)

#plt.axis("equal")
#plt.tight_layout()
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
