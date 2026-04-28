import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.interpolate import RegularGridInterpolator, splprep, splev
import imageio
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from affine import Affine
import contextily as ctx


# =====================================================
# PARAMETERS
# =====================================================
date_simu = "2026-04-20/15-27-52/"
which_flowline = "Ragna-Mariebreen"
point_fin = 8.4e3  # m
smooth = True
MAKE_GIF = True
GIF_FPS = 4


plot_obs = False

# ----------------------------
# Extent
# ----------------------------
extent = {
    "xmin": 553000,
    "xmax": 558700,
    "ymin": 8633000,
    "ymax": 8640500
}

# =====================================================
# PATHS
# =====================================================
simu_path = os.path.join("../outputs", date_simu)
out_dir = os.path.join(simu_path, "Plots/Bed_ice_type")
os.makedirs(out_dir, exist_ok=True)

flowline_file = os.path.join("../data", f"centerline_ragna_EM.csv")

# =====================================================
# LOAD FLOWLINE
# =====================================================
flow = pd.read_csv(flowline_file)
x_flow = flow["x"].values
y_flow = flow["y"].values

if smooth:
    tck, _ = splprep(np.vstack([x_flow, y_flow]), s=0, k=min(3, len(x_flow)-1))
    u = np.linspace(0, 1, 200)
    x_flow, y_flow = splev(u, tck)

dist_flow = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x_flow)**2 + np.diff(y_flow)**2))])
mask_flow = dist_flow <= point_fin
dist_km = dist_flow[mask_flow] / 1000.0

# =====================================================
# LOAD GPR_DATA
# =====================================================
if plot_obs:
    cts_obs_file = os.path.join("../data", f"CTS_centerline_ragna.csv")
    df_obs = pd.read_csv(cts_obs_file)
    x_obs = df_obs["x"].values
    cts_obs = df_obs[" y"].values
    long_pr_obs = 7.49e3
    dec = point_fin - long_pr_obs
    

# =====================================================
# LOAD MODEL OUTPUT
# =====================================================
ds = xr.open_dataset(os.path.join(simu_path, "output.nc"))

E = ds["E"]
Epmp = ds["E_pmp"]
thk = ds["thk"]
usurf = ds["usurf"]
topg = ds["topg"]
bg = topg
x = ds["x"].values
y = ds["y"].values
z = ds["z"].values
time = ds["time"].values

# Output_resolution
dx = x[1] - x[0]
dy = y[1] - y[0]

transform = Affine.translation(x.min(), y.min()) * Affine.scale(dx, dy)

# Dimensions (y, x)
ny = len(y)
nx = len(x)


ntime, nz = len(time), len(z)
vert_spacing = 4 
Nz=nz

zeta_edges = np.arange(Nz + 1) / Nz
zeta_edges = (zeta_edges / vert_spacing) * (1.0 + (vert_spacing - 1.0) * zeta_edges)
zeta_mid = 0.5 * (zeta_edges[:-1] + zeta_edges[1:])   # shape (nz,)



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
# ----------------------------
# Ensure y is monotonically increasing
# ----------------------------
reverse_y = False
if y[0] > y[-1]:
    y = y[::-1]
    reverse_y = True




# =====================================================
# ICE TYPE (0 = cold, 1 = tempered)
# =====================================================
ice_type_3d = np.zeros_like(E)
ice_type_3d[E >= Epmp] = 1


print( "ICE type 3D shape ", ice_type_3d.shape)


# =====================================================
# 🎨 COLORMAP ICE TYPE
# =====================================================
#cmap_ice = ListedColormap(["royalblue", "firebrick"])
cmap_ice = ListedColormap(["lightblue", "salmon"])
norm_ice = BoundaryNorm([-0.5, 0.5, 1.5], cmap_ice.N)

# =====================================================
# 🧊 VERTICAL SECTION (STATIC + GIF)
# =====================================================
frames = []
indices = range(ntime) if MAKE_GIF else [-1]
#indices = [-1] #range(ntime)

for it in indices:
    #type_bed = ice_type_3d[it, 1, :,:]
    type_bed = np.zeros_like(E[it,1,:,:])
    type_bed[E[it,1,:,:] >= Epmp[it,1,:,:]] = 1
    type_bed = np.where(mask, type_bed, np.nan)

    print( "Type bed shape ", type_bed.shape)

    #alpha = np.where(thk_last>0, 1.0, 0)
    fig,ax = plt.subplots(figsize=(10, 8))
    im=ax.pcolormesh(x, y, type_bed, shading='auto', cmap=cmap_ice, alpha=1, vmin=0, vmax=1)
    cbar=plt.colorbar(im, ax=ax)

    ctx.add_basemap(
    ax,
    crs="EPSG:32633",
    source=ctx.providers.Esri.WorldImagery
    )

    cbar.set_ticks([0.25, 0.75])
    cbar.ax.set_yticklabels(["Cold ice", "Temperate ice"])
    ax.set_title(f"Bed type – {nom_glacier}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    #ax.legend()
    #plt.axis("equal")
    #plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"type_bed_map_{time[it]}.png"), dpi=200)


    fname = os.path.join(out_dir, f"type_bed_map_{time[it]}.png")
    plt.savefig(fname, dpi=200)
    if not MAKE_GIF:
        plt.show()
    plt.close()

    if MAKE_GIF:
        frames.append(fname)



# =====================================================
# 🎞️ GIF
# =====================================================
if MAKE_GIF:
    gif_path = os.path.join(out_dir, "ice_type_vertical_section.gif")
    with imageio.get_writer(gif_path, mode="I", fps=GIF_FPS) as writer:
        for f in frames:
            writer.append_data(imageio.imread(f))
    print("GIF created:", gif_path)

# =====================================================
# Map frozen bed
# =====================================================

