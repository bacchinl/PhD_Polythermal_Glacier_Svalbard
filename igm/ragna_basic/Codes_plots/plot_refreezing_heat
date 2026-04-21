import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator, splprep, splev
import imageio

# ----------------------------
# Paths and output directory
# ----------------------------
date_simu = "2026-03-06/12-15-22/"


simu_path = os.path.join( "../outputs", date_simu)
out_dir = os.path.join("../outputs", date_simu, "Plots/temperature")
os.makedirs(out_dir, exist_ok=True)

which_flowline = "werenskiold_south"
flowline_file = os.path.join("../data", f"flowline_ragna.csv")

point_fin = 8.2e3 # en m
smooth = True

MAKE_GIF = True     # ← Mets False si tu veux juste le dernier plot
GIF_FPS   = 4       # images par seconde du GIF

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

T_var = ds['T']   
thk_var = ds["thk"]
usurf_var = ds['usurf']
topg_var = ds['topg']

x = ds['x'].values
y = ds['y'].values
z = ds['z'].values
time = ds['time']


print(time)
ntime = len(time)
nz = len(z)

# --- Interpolation de T le long de la flowline ---
# même logique que pour A3d_along_flow
T3d_along_flow = np.full((ntime, nz, len(x_flow)), np.nan)
thk_along_flow = np.full((ntime, len(x_flow)), np.nan)
usurf_along_flow = np.full((ntime, len(x_flow)), np.nan)
topg_along_flow = np.full((ntime, len(x_flow)), np.nan)

reverse_y = False
if y[0] > y[-1]:
    y = y[::-1]
    reverse_y = True

for it in range(ntime):
    T_t = T_var.isel(time=it).values  # shape (nz, ny, nx)
    thk_t = thk_var.isel(time=it).values
    usurf_t = usurf_var.isel(time=it).values
    topg_t = topg_var.isel(time=it).values

    if reverse_y:
        print("------------Y REVERSE-------------")
        T_t = T_t[:, ::-1, :]
        thk_t = thk_t[::-1, :]
        usurf_t = usurf_t[::-1, :]
        usurf_t = usurf_t[::-1, :]

    interp_T = RegularGridInterpolator((z, y, x), T_t, bounds_error=False, fill_value=np.nan)
    interp_thk = RegularGridInterpolator((y, x), thk_t, bounds_error=False, fill_value=np.nan)
    interp_usurf = RegularGridInterpolator((y, x), usurf_t, bounds_error=False, fill_value=np.nan)
    interp_topg = RegularGridInterpolator((y, x), topg_t, bounds_error=False, fill_value=np.nan)


    pts = np.vstack([y_flow, x_flow]).T
    thk_along_flow[it, :] = interp_thk(pts)
    usurf_along_flow[it, :] = interp_usurf(pts)
    topg_along_flow[it, :] = interp_topg(pts)   


    for k in range(nz):
        z_k = np.full_like(x_flow, z[k])
        pts3d = np.vstack([z_k, y_flow, x_flow]).T
        T3d_along_flow[it, k, :] = interp_T(pts3d)

print("Interpolation de T complétée.")
print("T3d_along_flow shape:", T3d_along_flow.shape)
print("thk_along_flow shape:", thk_along_flow.shape)

# ================================
# ⚖️ Moyenne verticale pondérée de T
# ================================

z_arr = z.copy()
if z_arr[0] > z_arr[-1]:
    z_arr = z_arr[::-1]

# Interfaces verticales (comme pour A)
z_if = np.empty(len(z_arr) + 1)
z_if[1:-1] = 0.5 * (z_arr[:-1] + z_arr[1:])
z_if[0]  = z_arr[0] - 0.5 * (z_arr[1] - z_arr[0])
z_if[-1] = z_arr[-1] + 0.5 * (z_arr[-1] - z_arr[-2])
layer_heights_ref = np.diff(z_if)
fractions = layer_heights_ref / np.sum(layer_heights_ref)

T_weighted_along_flow = np.full((ntime, len(x_flow)), np.nan)
T_simple_along_flow   = np.full((ntime, len(x_flow)), np.nan)

for it in range(ntime):
    T3d = T3d_along_flow[it, :, :]
    thk = thk_along_flow[it, :]

    layer_thickness = fractions[:, np.newaxis] * thk[np.newaxis, :]

    numer = np.nansum(T3d * layer_thickness, axis=0)
    denom = np.nansum(layer_thickness * (~np.isnan(T3d)), axis=0)
    valid = denom > 0
    T_weighted = np.full_like(thk, np.nan)
    T_weighted[valid] = numer[valid] / denom[valid]

    T_simple = np.nanmean(T3d, axis=0)

    T_weighted_along_flow[it, :] = T_weighted
    T_simple_along_flow[it, :]   = T_simple

print("Moyenne verticale pondérée de T complétée.")

# ================================
# 📈 Plot 1 – Température moyenne le long de la flowline (plusieurs années)
# ================================

years = np.array([int(t) for t in time.values])
years_to_plot = years[::2]

cmap = plt.colormaps.get('turbo_r')
norm = mcolors.Normalize(vmin=years_to_plot.min(), vmax=years_to_plot.max())

plt.figure(figsize=(10,6))
for yr in years_to_plot:
    idx = np.where(years == yr)[0][0]+1
    mask = dist_flow <= point_fin
    d_km = dist_flow[mask] / 1000.0

    plt.plot(d_km, T_simple_along_flow[idx, mask],
             color=cmap(norm(yr)), linewidth=2,
             label=str(yr))

#plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label="Année")
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = plt.colorbar(sm, ax=plt.gca(), label="Year")
plt.xlabel("Distance along the flowline (km)")
plt.ylabel("Mean temperature  (K)")
plt.title(f"Evolution of mean temperature along the flowline {which_flowline}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"T_profiles_along_{which_flowline}_flowline.png"), dpi=200)
plt.show()


# ================================
# 🌡️  Plot 3 – Coupe verticale réaliste de la température en mieux
# ================================
i_last = -1
year_i = time.values[i_last]
print(year_i)


T3d_last = T3d_along_flow[i_last, :, :]
usurf_last = usurf_along_flow[i_last, :]
topg_last = topg_along_flow[i_last, :]

# Découpage jusqu’à point_fin
mask = dist_flow <= point_fin
dist_crop = dist_flow[mask] / 1000.0
T3d_crop = T3d_last[:, mask]
usurf_crop = usurf_last[mask]
topg_crop = topg_last[mask]

# Interpolation verticale en altitude absolue
# (chaque niveau z est positionné entre bedrock et surface)
X = np.tile(dist_crop, (len(z), 1))
Z_abs = np.tile(z[:, np.newaxis], (1, len(dist_crop)))
Z_abs = topg_crop[np.newaxis, :] + Z_abs * (usurf_crop - topg_crop)[np.newaxis, :] / len(z)


print("Shape X :", X.shape, "Shape Z_abs :", Z_abs.shape,"Shape T3d_crop :", T3d_crop.shape,)

# --- PLOT ---
plt.figure(figsize=(11, 6))
levels = 40  # plus de niveaux => transitions plus douces
cont = plt.contourf(X, Z_abs, T3d_crop, levels=levels, cmap="coolwarm") #vmin = 270, vmax=273)
cbar = plt.colorbar(cont, label="Temperature (K)")

# Isothermes fines
plt.contour(X, Z_abs, T3d_crop, levels=10, colors="k", linewidths=0.3, alpha=0.4)

# Courbes de surface et de bedrock
plt.plot(dist_crop, usurf_crop, color="k", linewidth=1.5, label="Altitude")
plt.plot(dist_crop, topg_crop, color="k", linewidth=1.5, linestyle="--", label="Bedrock")

# Remplissage visuel de la glace
#plt.fill_between(dist_crop, topg_crop, usurf_crop, color="lightgray", alpha=0.3, label="Glace")

plt.xlabel("Distance along the flowline (km)")
plt.ylabel("Altitude (m a.s.l.)")
plt.title(f"Vertical distribution of temperature – {which_flowline} flowline, {year_i},final State")
plt.legend()
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"temperature_vertical_section_altitude_{which_flowline}_{years[i_last]}_Etat_final.png"), dpi=200)
plt.show()

# ================================
# 🌡️   Plot v3.2 – Coupe verticale réaliste avec option GIF
# ================================

frames = []  # pour stocker les chemins d’images si GIF activé

# Boucle sur TOUTES les années si GIF, sinon juste la dernière
indices = range(len(time)) if MAKE_GIF else [-1]

for i_last in indices:

    year_i = time.values[i_last]
    print(f"Plot {i_last} – Year:", year_i)

    # --- Extraction comme dans ton code ---
    T3d_last = T3d_along_flow[i_last, :, :]
    usurf_last = usurf_along_flow[i_last, :]
    topg_last = topg_along_flow[i_last, :]

    # Découpage jusqu’à point_fin
    mask = dist_flow <= point_fin
    dist_crop = dist_flow[mask] / 1000.0
    T3d_crop = T3d_last[:, mask]
    usurf_crop = usurf_last[mask]
    topg_crop = topg_last[mask]

    # Interpolation verticale
    X = np.tile(dist_crop, (len(z), 1))
    Z_abs = np.tile(z[:, np.newaxis], (1, len(dist_crop)))
    Z_abs = topg_crop[np.newaxis, :] + Z_abs * (usurf_crop - topg_crop)[np.newaxis, :] / len(z)

    # --- PLOT ---
    plt.figure(figsize=(11, 6))
    levels = 40
    cont = plt.contourf(X, Z_abs, T3d_crop, levels=levels, cmap="coolwarm", vmin=263, vmax=273.15)
    cbar = plt.colorbar(cont, label="Temperature (K)")

    # Isothermes + surfaces
    plt.contour(X, Z_abs, T3d_crop, levels=10, colors="k", linewidths=0.3, alpha=0.4)
    plt.plot(dist_crop, usurf_crop, color="k", linewidth=1.5, label="Altitude")
    plt.plot(dist_crop, topg_crop, color="k", linewidth=1.5, linestyle="--", label="Bedrock")

    plt.xlabel("Distance along the flowline (km)")
    plt.ylabel("Altitude (m a.s.l.)")
    plt.title(f"Vertical distribution of temperature – {which_flowline} flowline, {year_i}")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()

    # Nom du fichier (années en série si GIF)
    if MAKE_GIF == False and (i_last == 0 or i_last == len(time)-1) :
        fname = os.path.join(out_dir, f"temperature_vertical_section_altitude_{which_flowline}_{year_i}.png")
        plt.savefig(fname, dpi=200)
        plt.close()

    if MAKE_GIF:
        fname = os.path.join(out_dir, f"temperature_vertical_section_altitude_{which_flowline}_{year_i}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        frames.append(fname)  # stocke pour assemblage du GIF


# ================================
# 🎞️  Création du GIF (optionnel)
# ================================
if MAKE_GIF:
    gif_path = os.path.join(out_dir, f"temperature_vertical_section_altitude_{which_flowline}_ANIMATION.gif")
    print(f"Creating GIF: {gif_path}")

    with imageio.get_writer(gif_path, mode='I', fps=GIF_FPS) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))

    print("GIF terminé !")
