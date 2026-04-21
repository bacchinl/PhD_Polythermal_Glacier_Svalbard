import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator, splprep, splev

# =====================================================
# 🔧 PARAMÈTRES UTILISATEUR
# =====================================================
date_simu = "2025-11-18/09-02-38/"

simu_path = os.path.join( "~/../../datos3/Lucie/OUTPUT_IGM/WERENSKIOLBREEN/werensk-basic/outputs", date_simu)
out_dir = os.path.join("../../../../../../datos3/Lucie/OUTPUT_IGM/WERENSKIOLBREEN/werensk-basic/outputs", date_simu, "Plots/arrhenius")
os.makedirs(out_dir, exist_ok=True)


which_flowline = "werenskiold_south"
flowline_file = os.path.join("~/../../datos3/Lucie/DATA/WERENSKIOLBREEN", f"flowline_{which_flowline}_2022.csv")

point_fin = 5.7e3  # m
smooth = False

# =====================================================
# 📈 CHARGEMENT DE LA FLOWLINE
# =====================================================
flowline_df = pd.read_csv(flowline_file)
x_flow = flowline_df['x'].values
y_flow = flowline_df['y'].values

# Option : lisser la flowline
if smooth:
    tck, _ = splprep(np.vstack([x_flow, y_flow]), s=0, k=min(3, len(x_flow) - 1))
    n_samples = 200
    u_new = np.linspace(0, 1, n_samples)
    x_flow, y_flow = splev(u_new, tck)

# Distance cumulée
dist_flow = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x_flow)**2 + np.diff(y_flow)**2))])

# Point cible
idx_point_fin = np.argmin(np.abs(dist_flow - point_fin))
x_point_fin, y_point_fin = x_flow[idx_point_fin], y_flow[idx_point_fin]
print(f"Point à {point_fin/1000:.2f} km du départ : x={x_point_fin:.1f}, y={y_point_fin:.1f}")

# =====================================================
# 📦 CHARGEMENT DES DONNÉES DU MODÈLE
# =====================================================
ds = xr.open_dataset(os.path.join(simu_path, "output.nc"))

# Variables utiles
A_var = ds["arrhenius"]         # ton facteur d’Arrhenius (x, y, z, time)
thk_var = ds["thk"]             # épaisseur de glace (x, y, time)

x = ds["x"].values
y = ds["y"].values
z = ds["z"].values
time = ds["time"].values
years = np.array([int(t) for t in time])
ntime = len(years)
print(f"{ntime} pas de temps chargés : {years}")



# S’assurer que y soit croissant
reverse_y = False
if y[0] > y[-1]:
    y = y[::-1]
    reverse_y = True

# =====================================================
# 🧮 MOYENNE VERTICALE & STOCKAGE 3D
# =====================================================
A_bar = A_var.mean(dim="z")

# =====================================================
# 🔁 INTERPOLATION LE LONG DE LA FLOWLINE
# =====================================================
A_along_flow = np.full((ntime, len(x_flow)), np.nan)
thk_along_flow = np.full((ntime, len(x_flow)), np.nan)
A3d_along_flow = np.full((ntime, len(z), len(x_flow)), np.nan)

for it in range(ntime):
    # --- A moyen sur z ---
    A_t = A_bar.isel(time=it).values
    # --- A 3D (x,y,z) ---
    A3d_t = A_var.isel(time=it).values
    # --- épaisseur ---
    thk_t = thk_var.isel(time=it).values

    if reverse_y:
        A_t = A_t[::-1, :]
        A3d_t = A3d_t[:, ::-1, :]
        thk_t = thk_t[::-1, :]

    # Interpolation 2D pour A_bar et thk
    interp_A = RegularGridInterpolator((y, x), A_t, bounds_error=False, fill_value=np.nan)
    interp_thk = RegularGridInterpolator((y, x), thk_t, bounds_error=False, fill_value=np.nan)
    pts = np.vstack([y_flow, x_flow]).T
    A_along_flow[it, :] = interp_A(pts)
    thk_along_flow[it, :] = interp_thk(pts)

    # Interpolation 3D : boucle sur z
    for iz, zz in enumerate(z):
        interp_A3d = RegularGridInterpolator((y, x), A3d_t[iz, :, :],
                                             bounds_error=False, fill_value=np.nan)
        A3d_along_flow[it, iz, :] = interp_A3d(pts)

print("Interpolation du facteur d’Arrhenius (moyen + 3D) le long de la flowline terminée.")
print(f"A_along_flow shape: {A_along_flow.shape}, A3d_along_flow shape: {A3d_along_flow.shape}")

# =====================================================
# 🎨 1️⃣ PROFIL SPATIAL MOYEN SUR Z
# =====================================================
years_to_plot = years[::2]
cmap = plt.colormaps.get('turbo_r')
norm = mcolors.Normalize(vmin=years_to_plot.min(), vmax=years_to_plot.max())

plt.figure(figsize=(10, 6))
for yr in years_to_plot:
    idx = np.where(years == yr)[0][0]
    dist_mask = dist_flow <= point_fin
    plt.plot(dist_flow[dist_mask] / 1000.0,
             A_along_flow[idx, dist_mask],
             color=cmap(norm(yr)),
             linewidth=2,
             label=str(yr))

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=plt.gca(), label="Année")
plt.xlabel("Distance along the flowline (km)")
plt.ylabel("Arrhenius factor (MPa$^{-3}$ a$^{-1}$)")
plt.title(f"Évolution of Arrhenius factor (z_mean) – {which_flowline}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"arrhenius_profiles_along_{which_flowline}.png"), dpi=200)
plt.show()

# =====================================================
# 🎨 2️⃣ COUPE VERTICALE (z vs distance) JUSQU’À point_fin
# =====================================================
i_last = -1
A3d_last = A3d_along_flow[i_last, :, :]
thk_last = thk_along_flow[i_last, :]

# Masquer au-delà du point_fin
dist_mask = dist_flow <= point_fin
dist_crop = dist_flow[dist_mask] / 1000.0  # en km
A3d_crop = A3d_last[:, dist_mask]
thk_crop = thk_last[dist_mask]

# niveaux verticaux normalisés (sigma)
nz = len(z)
sigma = np.linspace(0, 1, nz)  # 0 = base, 1 = surface

# grille pour les distances
n_flow_crop = len(dist_crop)
X = np.tile(dist_crop, (nz, 1))

# créer la grille verticale réelle (chaque colonne de z dépend de thk locale)
Z_real = np.zeros_like(X)
for i in range(n_flow_crop):
    Z_real[:, i] = sigma * thk_crop[i]  # vertical stretching

# Masquer au-dessus de la glace (inutile ici car déjà borné)
A_plot = np.copy(A3d_crop)
A_plot[Z_real > thk_crop] = np.nan

# ---- Plot ----
plt.figure(figsize=(11, 6))
pcm = plt.pcolormesh(X, Z_real, A_plot, cmap="coolwarm", shading="auto")
plt.colorbar(pcm, label="Facteur d’Arrhenius A (MPa⁻³ a⁻¹)")

# Tracer la surface du glacier
plt.plot(dist_crop, thk_crop, color="k", linewidth=2.5, label="Surface du glacier")

# Remplir la glace en gris clair pour l’aspect réaliste
plt.fill_between(dist_crop, 0, thk_crop, color="lightgray", alpha=0.2)

plt.xlabel("Distance le long de la flowline (km)")
plt.ylabel("Profondeur dans la glace (m)")
plt.title(f"Coupe verticale du facteur d’Arrhenius (réelle) – année {years[i_last]}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"arrhenius_vertical_real_{which_flowline}_{years[i_last]}.png"), dpi=200)
plt.show()
























# =====================================================
# 🗺️ 3️⃣ OPTIONNEL : CHAMP 2D MOYENNÉ + FLOWLINE
# =====================================================
A_last = A_bar.isel(time=i_last).values
if reverse_y:
    A_last = A_last[::-1, :]
    y_plot = y
else:
    y_plot = y
thk_2d_last = thk_var.isel(time=i_last).values
if reverse_y:
    thk_2d_last = thk_2d_last[::-1, :]


alpha = np.where(thk_2d_last > 0, 1.0, 0.0)
plt.figure(figsize=(10, 8))
plt.pcolormesh(x, y_plot, A_last, shading='auto', cmap='coolwarm', alpha=alpha)
plt.colorbar(label="A ( z mean )")
plt.plot(x_flow, y_flow, 'r-', linewidth=2.5, label="Flowline")
plt.scatter(x_point_fin, y_point_fin, color='green', edgecolor='black', s=20, label=f"{point_fin/1000:.1f} km")
plt.title(f"Arrhenius factor mean on z – year {years[i_last]}")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"arrhenius_field_flowline_{which_flowline}_{years[i_last]}.png"), dpi=200)
plt.show()


alpha = np.where(thk_2d_last > 0, 1.0, 0.0)
plt.figure(figsize=(10, 8))
plt.pcolormesh(x, y_plot, thk_2d_last, shading='auto', cmap='Blues', alpha=alpha)
plt.colorbar(label="Thickness (m)")
plt.plot(x_flow, y_flow, 'r-', linewidth=2.5, label="Flowline")
plt.scatter(x_point_fin, y_point_fin, color='green', edgecolor='black', s=20, label=f"{point_fin/1000:.1f} km")
plt.title(f"Thickness year {years[i_last]}")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"thickness_flowline_{which_flowline}_{years[i_last]}.png"), dpi=200)
plt.show()

