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
date_simu = "2026-04-20/15-27-52/"
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
out_dir = os.path.join(simu_path, "Plots/ice_type")
os.makedirs(out_dir, exist_ok=True)

flowline_file = os.path.join("../data", f"centerline_ragna_EM_mod.csv")

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
dist_km = dist_km[::-1]

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
# INTERPOLATION ALONG FLOWLINE
# =====================================================
E3d = np.full((ntime, nz, len(x_flow)), np.nan)
Epmp3d = np.full_like(E3d, np.nan)
thk_f = np.full((ntime, len(x_flow)), np.nan)
usurf_f = np.full_like(thk_f, np.nan)
topg_f = np.full_like(thk_f, np.nan)

for it in range(ntime):

    interp_E = RegularGridInterpolator((z, y, x), E.isel(time=it).values)
    interp_Epmp = RegularGridInterpolator((z, y, x), Epmp.isel(time=it).values)
    interp_thk = RegularGridInterpolator((y, x), thk.isel(time=it).values)
    interp_usurf = RegularGridInterpolator((y, x), usurf.isel(time=it).values)
    interp_topg = RegularGridInterpolator((y, x), topg.isel(time=it).values)

    pts2d = np.vstack([y_flow, x_flow]).T
    thk_f[it] = interp_thk(pts2d)
    usurf_f[it] = interp_usurf(pts2d)
    topg_f[it] = interp_topg(pts2d)

    for k in range(nz):
        pts3d = np.vstack([np.full_like(x_flow, z[k]), y_flow, x_flow]).T
        E3d[it, k] = interp_E(pts3d)
        Epmp3d[it, k] = interp_Epmp(pts3d)

print("Interpolation completed")

# =====================================================
# ICE TYPE (0 = cold, 1 = tempered)
# =====================================================
ice_type_3d = np.zeros_like(E3d)
ice_type_3d[E3d >= Epmp3d] = 1

# =====================================================
# COLD LAYER THICKNESS
# =====================================================
# vertical layer thickness (sigma-like)
z_if = np.zeros(nz + 1)
z_if[1:-1] = 0.5 * (z[:-1] + z[1:])
z_if[0] = z[0] - (z[1] - z[0]) / 2
z_if[-1] = z[-1] + (z[-1] - z[-2]) / 2
dz = np.diff(z_if)
fractions = dz / dz.sum()

cold_thickness = np.full((ntime, len(x_flow)), np.nan)

for it in range(ntime):
    layer_thk = fractions[:, None] * thk_f[it]
    cold_mask = ice_type_3d[it] == 0
    cold_thickness[it] = np.nansum(layer_thk * cold_mask, axis=0)

# =====================================================
# 📈 PLOT 1 – Cold layer thickness vs time (flowline)
# =====================================================
plt.figure(figsize=(10, 6))

years = np.array([int(t) for t in time])
years = np.unique(years)

print("--- YEAR---", years)
cmap = cm.get_cmap("turbo")
colors = cmap(np.linspace(0, 1, len(years)))


for i, yr in enumerate(years):
    plt.plot(dist_km, cold_thickness[i, mask_flow], color=colors[i], alpha=0.8,label=str(yr))

plt.plot(
    dist_km,
    np.nanmean(cold_thickness[:, mask_flow], axis=0),
    color="k",
    linewidth=2,
    label="Mean"
)

plt.xlabel("Distance along flowline (km)")
plt.ylabel("Cold layer thickness (m)")
plt.title("Cold ice layer thickness along flowline (all years)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "cold_layer_thickness_along_flowline.png"), dpi=200)
plt.show()

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

    ice_crop = ice_type_3d[it, :, mask_flow]
    us = usurf_f[it, mask_flow]
    bg = topg_f[it, mask_flow]
    thk = thk_f[it, mask_flow]
    type_bed = ice_type_3d[it, 0, :]   


    X = np.tile(dist_km, (nz, 1))
    Z = bg[None, :] + zeta_mid[:, None] * (us - bg)[None, :] #/ nz
    #Z = bg[None, :] + zeta_mid[:, None] * thk[None, :]
    if ice_crop.shape != X.shape:
        ice_crop = ice_crop.T

    if E3d.shape != X.shape:
         E3d = E3d.T



    plt.figure(figsize=(11, 5))

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
    cbar.ax.set_yticklabels(["Cold ice", "Temperate ice"])

    z_min = np.min(Z)

    # Remplir la zone sous la surface du glacier en marron
    plt.fill_between(
        dist_km, bg, z_min,
        color="saddlebrown",
        alpha=0.8,
        zorder=2,
        label="Bedrock"
    )

    plt.xlabel("Distance along flowline (km)")
    plt.ylabel("Altitude (m a.s.l.)")
    plt.title(f"Ice type vertical section – {which_flowline} – without refreezing")
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

