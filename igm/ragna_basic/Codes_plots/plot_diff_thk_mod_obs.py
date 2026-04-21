import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin, xy
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import generic_filter

simu_path = "../outputs/2025-11-11/12-32-20/"
out_dir = f"{simu_path}/Plots/"
os.makedirs(out_dir, exist_ok=True)

PLOT_EPAISSEURS = True
zomm_dis = 2000

year = 2021
SHOW = False

# --- Charger les données ---
# Charger le modèle (NetCDF)
ds = xr.open_dataset(f"{simu_path}/output.nc")
print(ds)
ds_in = xr.open_dataset("../data/input.nc")
icemask = ds_in["icemask"]



# Sélectionner la variable d'épaisseur (remplace 'epaisseur' par le vrai nom si différent)
x = ds['x'].values
y = ds['y'].values
print(x[0], x[-1])
print(y[0], y[-1])

thk = ds['thk']   # (time, y, x)
time = ds['time']
for i in range(0, len(time)):
    year_i = int(time[i].values)
    if year_i == year :
        print(f"Year {year} found")
        thk_var=thk.isel(time=i)
        thk_var_flipped = thk_var[::-1, :] 

# Calculer la taille des pixels (en supposant une grille régulière)
dx = np.abs(x[1] - x[0])
dy = np.abs(y[1] - y[0])

# Attention : y peut être décroissant
if y[0] > y[-1]:
    dy = np.abs(y[1] - y[0])
    src_transform = from_origin(x.min(), y[0], dx, dy)  # y[0] = coin supérieur gauche
    print("nord vers sud")
else:
    src_transform = from_origin(x.min(), y.max(), dx, dy)
    print("sud vers nord")
    print("Xmin : ", x.min(), "Xmax : ", x.max())
    print("ymin : ", y.min(), "ymax : ", y.max())
    
# Charger le TIFF observé
with rasterio.open(f"../../../DATA/From_Valentin/thk_ref_{year}.tif") as src :
    thk_obs = src.read(1)
    print(src.crs)
    obs_meta = src.meta
    dst_transform = src.transform
    dst_crs = src.crs
    dst_shape = (src.height, src.width)



thk_mod = thk_var.values.squeeze()
thk_mod_flipped = thk_mod[::-1, :]

thk_mod_reproj = np.empty_like(thk_obs)

reproject(
    source=thk_var_flipped,
    destination=thk_mod_reproj,
    src_transform=src_transform,
    src_crs="EPSG:32633",
    dst_transform=dst_transform,
    dst_crs=dst_crs,
    resampling=Resampling.bilinear
)


# --- Calcul de la différence ---
diff = thk_mod_reproj - thk_obs.squeeze() # .squeeze() pour retirer la dimension de bande
diff_masked = np.where(icemask.values == 1, diff, np.nan)
diff_glacier = diff_masked[~np.isnan(diff_masked)]

height, width = thk_mod_reproj.shape
x0, y0 = xy(dst_transform, 0, 0)        # coin supérieur gauche
x1, y1 = xy(dst_transform, height-1, width-1)  # coin inférieur droit

extent = [x0, x1, y1, y0]

############################ STATS ###########################
#q5, q95 = np.nanpercentile(diff_glacier, [5, 95])
diff_filtered = diff_glacier[(diff_glacier <= 3e38)]
#diff_filtered = diff_glacier[(diff_glacier >= q5) & (diff_glacier <= q95)]
q5, q95 = np.nanpercentile(diff_filtered, [5, 95])
mean_val = np.mean(diff_filtered)
std_val = np.std(diff_filtered)


print("Nombre de pixels (non-NaN) :", np.sum(~np.isnan(diff_masked)))
print(f"Quantile 5%: {q5:.2f}, Quantile 95%: {q95:.2f}")
print(f"Moyenne : {mean_val:.2f} m")
print(f"Écart-type global : {std_val:.2f} m")




if PLOT_EPAISSEURS :
    alpha_mod = np.where(thk_mod_reproj >= 0.1, 1.0, 0.0)
    alpha_obs = np.where(thk_obs.squeeze() >= 0.1, 1.0, 0.0)
    plt.figure(figsize=(8,6))
    plt.imshow(thk_obs.squeeze(), cmap="RdYlBu_r", vmin=0, vmax=400, alpha=alpha_obs)
    plt.colorbar(label="Thickness (m)")
    plt.title(f"thickness (observation){year}")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.savefig(f"{out_dir}/observed_thickness_{year}.png", dpi=150)
    if SHOW :
        plt.show()

    plt.figure(figsize=(8,6))
    plt.imshow(thk_mod_reproj, cmap="RdYlBu_r", vmin=0, vmax=400, alpha=alpha_mod)
    plt.colorbar(label="Thickness (m)")
    plt.title(f"thickness (model) {year}")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.savefig(f"{out_dir}/modelled_thickness_{year}.png", dpi=150)
    if SHOW :    
        plt.show()


plt.figure(figsize=(8,6))
plt.imshow(diff_masked, cmap="RdBu_r", vmin=-50, vmax=50, extent=extent)
plt.colorbar(label="Thickness difference (m)")
plt.title(f"Thickness difference (model-obs) {year}, σ = {std_val:.2f}m, mean={mean_val:.2f} m")
plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
#plt.xlim(504000, 514000)  # exemple en mètre
#plt.ylim(8551000, 8460000)
plt.savefig(f"{out_dir}/difference_thickness_{year}.png", dpi=150)
if SHOW :
    plt.show()


plt.figure(figsize=(7,5))
plt.hist(diff_filtered, bins=50, color="skyblue", edgecolor="k")
plt.xlabel("Thickness difference (m)")
plt.ylabel("Pixel number")
plt.title(f"Repartition of the thickness difference (quantile 5-95%) {year}")
plt.savefig(f"{out_dir}/repartition_difference_{year}.png", dpi=150)
if SHOW :
    plt.show()

std_map = generic_filter(diff_masked, np.std, size=3, mode='constant', cval=np.nan)



plt.figure(figsize=(8,6))
plt.imshow(std_map, cmap="magma", extent=extent, vmin=0, vmax =5)
plt.colorbar(label="standard deviation (m)")
plt.title(f"Map of the standard deviation, {year}")
plt.savefig(f"{out_dir}/standard_deviation_thickness_{year}.png", dpi=150)
if SHOW :
    plt.show()


# --- Sauvegarde du résultat ---
#diff.rio.to_raster("diff_epaisseur_2021.tif")

#print("✅ Différence calculée et sauvegardée dans 'diff_epaisseur_2021.tif'")
