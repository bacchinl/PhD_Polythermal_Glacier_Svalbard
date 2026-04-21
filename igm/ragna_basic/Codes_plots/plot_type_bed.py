import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.interpolate import RegularGridInterpolator, splprep, splev
import imageio

# =====================================================
# PARAMETERS
# =====================================================
date_simu = "2026-04-15/12-41-12/"
which_flowline = "Ragna-Mariebreen"
point_fin = 8.4e3  # m
smooth = True
MAKE_GIF = True
GIF_FPS = 4


plot_obs = False

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

x = ds["x"].values
y = ds["y"].values
z = ds["z"].values
time = ds["time"].values

ntime, nz = len(time), len(z)
vert_spacing = 4 
Nz=nz

zeta_edges = np.arange(Nz + 1) / Nz
zeta_edges = (zeta_edges / vert_spacing) * (1.0 + (vert_spacing - 1.0) * zeta_edges)
zeta_mid = 0.5 * (zeta_edges[:-1] + zeta_edges[1:])   # shape (nz,)


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

for it in indices:
    type_bed = ice_type_3d[it, 0, :,:]
    print( "Type bed shape ", type_bed.shape)

    #alpha = np.where(thk_last>0, 1.0, 0)
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(x, y, type_bed, shading='auto', cmap=cmap_ice, alpha=1, vmin=0, vmax=1)
    plt.colorbar(label="sliding coefficient (km MPa$^{-3}$ a$^{-1}$)")


    plt.title(f"Bed type – Year {time[it]}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"type_bed_map_{time[it]}.png"), dpi=200)
    plt.show()

    X = np.tile(dist_km, (nz, 1))
    Z = bg[None, :] + zeta_mid[:, None] * (us - bg)[None, :] #/ nz

    if ice_crop.shape != X.shape:
        ice_crop = ice_crop.T

    if E3d.shape != X.shape:
         E3d = E3d.T



    plt.figure(figsize=(11, 6))

    plt.contourf(
        X, Z, ice_crop,
        levels=[-0.5, 0.5, 1.5],
        cmap=cmap_ice,
        norm=norm_ice
    )
    if plot_obs:
        plt.plot(x_obs, cts_obs, "k--", lw=2, label="CTS (Mannerfelt et al. 2024)")

    plt.plot(dist_km, us, "k", lw=1.5)
    plt.plot(dist_km, bg, "k--", lw=1.2)

    #plt.contour(
    #    X, Z,
    #    E3d[it, :, mask_flow] - Epmp3d[it, :, mask_flow],
    #    levels=[0], colors="k", linewidths=1.2
    #)

    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(["Cold ice", "Tempered ice"])

    plt.xlabel("Distance along flowline (km)")
    plt.ylabel("Altitude (m a.s.l.)")
    plt.title(f"Ice type vertical section – {which_flowline} – {time[it]}")
    plt.grid(True, linestyle=":")
    plt.tight_layout()

    fname = os.path.join(out_dir, f"ice_type_section_{time[it]}.png")
    plt.savefig(fname, dpi=200)
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

