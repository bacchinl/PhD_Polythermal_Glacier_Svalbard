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
from affine import Affine
import rioxarray as rxr
import contextily as ctx

#----------------------------
# Paths and output directory
# ----------------------------
date_simu_with_r = "2026-04-20/12-11-45"
date_simu_without_r = "2026-04-20/15-27-52"




simu_path_yes = os.path.join( "../outputs", date_simu_with_r)
simu_path_no = os.path.join( "../outputs", date_simu_without_r)

out_dir = os.path.join("../outputs", date_simu_with_r, "Plots/Comparison_velocity")
os.makedirs(out_dir, exist_ok=True)

which_flowline = "mette"
flowline_file = os.path.join("../data/", f"centerline_ragna_EM_mod.csv")


point_fin = 7.4e3 # en m
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
def open_ds(simu_path):
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
    return x,y,u_var,v_var,years,ntime

x,y,u_var_y,v_var_y,years, ntime = open_ds(simu_path_yes)
x_n,y_n,u_var_n,v_var_n,years_n,ntime_n  = open_ds(simu_path_no)


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
# Interpolate velocity along flowline
# ----------------------------
speed_along_flow_y = np.full((ntime, len(x_flow)), np.nan)
speed_along_flow_n = np.full((ntime, len(x_flow)), np.nan)

for it in range(ntime):
    u_t_y = u_var_y.isel(time=it).values
    v_t_y = v_var_y.isel(time=it).values

    u_t_n = u_var_n.isel(time=it).values
    v_t_n = v_var_n.isel(time=it).values
    if reverse_y:
        u_t_y = u_t_y[::-1, :]
        v_t_y = v_t_y[::-1, :]
        u_t_n = u_t_n[::-1, :]
        v_t_n = v_t_n[::-1, :]

    speed_t_y = np.sqrt(u_t_y**2 + v_t_y**2)
    speed_t_n = np.sqrt(u_t_n**2 + v_t_n**2)

    interp_y = RegularGridInterpolator((y, x), speed_t_y,
                                     bounds_error=False, fill_value=np.nan)
    interp_n = RegularGridInterpolator((y, x), speed_t_n,
                                     bounds_error=False, fill_value=np.nan)

    pts = np.vstack([y_flow, x_flow]).T
    speed_along_flow_y[it, :] = interp_y(pts)
    speed_along_flow_n[it, :] = interp_n(pts)


print("Velocity interpolation along the flowline completed. Shape:", speed_along_flow_y.shape)




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
    dist_mask = dist_flow <= point_fin
    diff_vel = speed_along_flow_y[idx, dist_mask]-speed_along_flow_n[idx, dist_mask]
    

    plt.plot(dist_flow[dist_mask] / 1000.0,
             diff_vel,
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

def calcul_speed_last(u_var,v_var):
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
    u_dir = u_last / np.where(speed_last == 0, np.nan, speed_last)
    v_dir = v_last / np.where(speed_last == 0, np.nan, speed_last)
    
    return speed_last, u_dir, v_dir, y_plot

speed_last_y, u_dir, v_dir, y_plot = calcul_speed_last(u_var_y,v_var_y)
speed_last_n, u_dir_n, v_dir_n, y_plot_n = calcul_speed_last(u_var_n,v_var_n)

diff_speed_last = speed_last_y-speed_last_n

ix = np.where((x >= extent["xmin"]) & (x <= extent["xmax"]))[0]
iy = np.where((y_plot >= extent["ymin"]) & (y_plot <= extent["ymax"]))[0]

x_sub = x[ix]
y_sub = y_plot[iy]

diff_speed_sub = diff_speed_last[np.ix_(iy, ix)]
u_dir_sub = u_dir[np.ix_(iy, ix)]
v_dir_sub = v_dir[np.ix_(iy, ix)]




# optional: smooth flowline using spline
tck, _ = splprep(np.vstack([x_flow, y_flow]), s=0, k=min(3, len(x_flow)-1))
n_samples = 200
u_new = np.linspace(0, 1, n_samples)
x_flow_smooth, y_flow_smooth = splev(u_new, tck)


#alpha = np.where(speed_last > 0, 1.0, 0.0)

plt.figure(figsize=(10, 8))
plt.pcolormesh(x, y_plot, diff_speed_last, shading='auto', cmap='bwr', vmin=-1, vmax=1)
plt.colorbar(label="Velocity difference (m/an)")

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
#plt.scatter(x_point_fin, y_point_fin, color='green', edgecolor='black', s=20, zorder=5, label=f"Point @ {point_fin/1000:.1f} km")

plt.title(f"Velocity field and {which_flowline} flowline – Year {years[i_last]}")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"comparaison_velocity_field_flowline_{which_flowline}_{years[i_last]}.png"), dpi=200)
plt.show()



fig,ax = plt.subplots(figsize=(10, 8))

# ---- Background velocity amplitude ----
im = ax.pcolormesh(x_sub, y_sub, diff_speed_sub, shading='auto',
               cmap='RdBu_r', vmin=-0.5, vmax=0.5)
cbar = plt.colorbar(im, ax=ax)

# ---- Grid ----
X, Y = np.meshgrid(x_sub, y_sub)


# -------------------------------
# STREAMPLOT
# -------------------------------
ax.streamplot(
    X, Y,
    u_dir_sub, v_dir_sub,            # direction field
    density=2,             # increase for more curves
    color='k',               # streamline color
    linewidth=0.8,           # line thickness
    arrowsize=0.8            # small arrowheads on the streamlines
)

ax.set_xlim(extent["xmin"], extent["xmax"])
ax.set_ylim(extent["ymin"], extent["ymax"])

ctx.add_basemap(
    ax,
    crs="EPSG:32633",
    source=ctx.providers.Esri.WorldImagery
)

# ---- Plot flowline ----
#plt.plot(x_flow_smooth, y_flow_smooth,         'r-', linewidth=2.5, label="Flowline")

# ---- Point_fin ----
#plt.scatter(x_point_fin, y_point_fin,            color='green', edgecolor='black', s=20, zorder=5,label=f"Point @ {point_fin/1000:.1f} km")
ax.set_xlim(extent["xmin"], extent["xmax"])
ax.set_ylim(extent["ymin"], extent["ymax"])

ax.set_title(f"Velocity difference field (with - without refreezing) - Ragna-Mariebreen ", fontsize = 14)
ax.set_xlabel("Longitude", fontsize = 14)
ax.set_ylabel("Latitude", fontsize = 14)
ax.legend()

cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
cbar.set_label("Velocity difference (m/a)", fontsize=16)
cbar.ax.tick_params(labelsize=12)
#plt.axis("equal")
ax.set_autoscale_on(False)
#plt.tight_layout()
plt.savefig(os.path.join(out_dir,
            f"comparaison_velocity_field_flowline_{which_flowline}_{years[i_last]}_streamlines.png"),
            dpi=200)
plt.show()


# ----------------------------
# Save flowline for reuse
# ----------------------------
#flowline_saved = pd.DataFrame({'x': x_flow, 'y': y_flow})
#flowline_saved.to_csv(os.path.join(out_dir, f"flowline_{which_flowline}_2022_saved.csv"), index=False)
#print("Flowline saved to CSV for reuse.")
