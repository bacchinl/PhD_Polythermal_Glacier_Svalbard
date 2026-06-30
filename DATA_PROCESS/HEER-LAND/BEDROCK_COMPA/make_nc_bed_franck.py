import os
import rasterio
from rasterio.merge import merge
from rasterio.features import geometry_mask
from rasterio.errors import RasterioIOError
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import matplotlib.pyplot as plt
from rasterio.transform import array_bounds


exclude = [  333, 355, 940] ## 333 and 355 in nordenskioldland and 940 is just fucked up
plot_tuile = False
# --- chemins ---
raster_dir = os.path.expanduser("~/PhD_Lucie/DATA/BEDROCK/Thomas_Frank/Topg/RGI-07_topg/topg_final/")
shp_path = "~/PhD_Lucie/DATA/Maps/Shapefiles_regions/mask_heer_land.shp"

# --- charger shapefile ---
gdf = gpd.read_file(shp_path)

# s'assurer du CRS (EPSG:32633)
gdf = gdf.to_crs("EPSG:32633")
mask_geom = gdf.unary_union  # géométrie unique

selected_files = []

mins = []
maxs = []

# --- boucle sur tous les indices possibles ---
for i in range(1, 1616):
     
    if i in exclude:
        continue
    
    fname = f"RGI60-07.{i:05d}_topg.tif"
    fpath = os.path.join(raster_dir, fname)

    # 1) check existence explicite
    if not os.path.exists(fpath):
        continue

    try:
        with rasterio.open(fpath) as src:
            # ici ton test spatial
            raster_bounds = box(*src.bounds)

            if raster_bounds.intersects(mask_geom):
                selected_files.append(fpath)
                print(f"{fname} -> CRS: {src.crs}")
                # --- lecture rapide pour stats ---
                data = src.read(1, masked=True)

                mins.append(np.nanmin(data))
                maxs.append(np.nanmax(data))                
                
                # =========================
                # PLOT PAR TUILE
                # =========================
                if plot_tuile:
                    plt.figure(figsize=(6, 5))

                    im = plt.imshow(
                        data,
                        cmap="terrain",
                        vmin=0,
                        vmax=600
                    )

                    plt.title(f"RGI60-07.{i:05d}")
                    plt.colorbar(im, label="Elevation (m)")

                    plt.tight_layout()
                    plt.show()

    except (FileNotFoundError, RasterioIOError, Exception) as e:
        # 2) fichier cassé / illisible / GDAL error → skip
        print(f"skip {fname} -> {e}")
        continue

print(f"{len(selected_files)} rasters sélectionnés")

# --- merge des rasters sélectionnés ---
plt.figure()

plt.plot(mins, label="min", linewidth=1)
plt.plot(maxs, label="max", linewidth=1)

plt.title("Min / Max values per selected raster")
plt.xlabel("Raster index (selected order)")
plt.ylabel("Elevation")
plt.legend()
plt.show()


srcs = [rasterio.open(fp) for fp in selected_files]


mosaic, out_transform = merge(srcs)



plt.figure(figsize=(12, 10))

im = plt.imshow(
    mosaic[0],
    cmap="terrain",
    origin="upper",
    vmin=0,
    vmax=600
)

plt.title("mosaic (masked)")
plt.colorbar(im, label="Elevation (m)")
plt.tight_layout()
plt.show()






out_meta = srcs[0].meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_transform
})



out_path = os.path.join(".", "mosaic_heer_land_franck.tif")

with rasterio.open(out_path, "w", **out_meta) as dest:
    dest.write(mosaic)

# fermer les fichiers
for s in srcs:
    s.close()

print("Mosaic créé :", out_path)






