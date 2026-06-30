import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.interpolate import griddata, RegularGridInterpolator
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
import xarray as xr
from scipy.ndimage import gaussian_filter

#### MAME A PRIMARY NCDF OF THE BEDROCK TOPOGRAPHY OF {glacier_name} WITH GPR DATA


dx=20
dy=dx

glacier_name = "kroppbreen"

#please, epsg 32633
if glacier_name == "mettebreen" : 
    x_min, x_max, y_min, y_max= [549900, 558800, 8631600, 8642200] #please, epsg 32633

elif glacier_name == "kroppbreen":
    x_min, x_max, y_min, y_max= [553300, 559700, 8644100, 8654200]

elif glacier_name == "vallakrabreen":
    x_min, x_max, y_min, y_max= [546100, 554000, 8639300, 8648700]

elif glacier_name == "edvardbreen":
    x_min, x_max, y_min, y_max= [553400, 566400, 8639700, 8655200]







extent = [x_min, x_max, y_min, y_max]

# Calcul du nombre de cellules
#nx = int(np.ceil((x_max - x_min) / dx))
#ny = int(np.ceil((y_max - y_min) / dy))

poly_subextent = Polygon([
    (x_min, y_min),
    (x_min, y_max),
    (x_max, y_max),
    (x_max, y_min)
])



biharmonic=False
sigma = 2.5 #gaussian filter smoothing

# ----------------- Charger le fichier CSV
data_dir = os.path.expanduser(
    "~/PhD_Lucie/DATA/GPR/Mannerfelt/thickness_cts_points_csv"
)
csv_pattern = os.path.join(
    data_dir,
    f"thickness_cts_points_{glacier_name}*.csv"
)

csv_files = sorted(glob.glob(csv_pattern))


#df = pd.read_csv("~/PhD_Lucie/DATA/GPR/Mannerfelt/thickness_cts_points_csv/.csv", header=0)
#print(df.columns)


#df = df[(df[" Easting"] > 4 * 1e5) & (df["Depth CTS re-calculated"] <= 1000)]
x,y,z,surf = [],[],[],[]

for csv_file in csv_files:

    print("\n" + "=" * 60)
    print(f"Processing: {os.path.basename(csv_file)}")

    # =====================================================
    # EXTRACT RADAR LINE NAME
    # =====================================================

    radar_line = (
        os.path.basename(csv_file)
        .replace("thickness_cts_points_", "")
        .replace(".csv", "")
    )

    print(f"Radar line: {radar_line}")
    # =====================================================
    # LOAD CSV
    # =====================================================

    columns_to_keep = [
        "radar_key",
        "distance",
        "easting",
        "northing",
        "elevation",
        "bed_elevation"
    ]

    gpr_df = pd.read_csv(
        csv_file,
        usecols=columns_to_keep
    )

    gpr_df_trace = gpr_df[
        gpr_df["radar_key"] == radar_line
    ].copy()


    # Extraire les colonnes
    x_trace = gpr_df_trace["easting"]
    y_trace = gpr_df_trace["northing"]
    z_trace = gpr_df_trace["bed_elevation"]
    surf_trace = gpr_df_trace["elevation"]

    x.append(x_trace)
    y.append(y_trace)
    z.append(z_trace)
    surf.append(surf_trace)



x = pd.concat(x).values
y = pd.concat(y).values
z = pd.concat(z).values
surf = pd.concat(surf).values


print("bed elev between ", np.min(z), " & ", np.max(z))

x_min_e, y_min_e, x_max_e, y_max_e = np.min(x), np.max(x), np.min(y), np.max(y)
print("Extent GPR : ", x_min_e, y_min_e, x_max_e, y_max_e)


# -------------------  charger mask

path_shp = "~/PhD_Lucie/DATA/GLACIER_OUTLINES/CryoClim_GAO_SJ_1990/"
shp_path = os.path.join(path_shp, "CryoClim_GAO_SJ_1990.shp")

gdf = gpd.read_file(shp_path)
gdf_one = gdf[gdf["NAME"] == "Ragnamariebreen"]


gdf_multi = gdf[gdf.geometry.intersects(poly_subextent)]
shapes = [(geom, 1) for geom in gdf_multi.geometry]


mask_geom = gdf_multi.unary_union







plt.scatter(x, y,c=z, s=2)
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="bed depth")
plt.show()





# Création du plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(x, y, z, c=z)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Profondeur")

plt.colorbar(sc, label="Profondeur")
plt.show()



 ####### <interp bed interp

#xi = np.arange(x.min(), x.max(), dx)
#yi = np.arange(y.min(), y.max(), dx)


xi = np.arange(x_min, x_max, dx)
yi = np.arange(y_min, y_max, dx)
Xi, Yi = np.meshgrid(xi, yi)
nx =len(xi)
ny = len(yi)

Zi = griddata(
    (x, y),
    z,
    (Xi, Yi),
    method="linear"   # 'nearest' ou 'cubic' possibles
)


Surfi = griddata(
    (x, y),
    surf,
    (Xi, Yi),
    method="linear"   # 'nearest' ou 'cubic' possibles
)



Zi = gaussian_filter(Zi, sigma=sigma)

mask = np.zeros(Zi.shape, dtype=bool)

transform_sub = from_origin(x_min, y_max, dx, dy)
print(transform_sub)

mask_tot_ice = rasterize(
    shapes=shapes,
    out_shape=(ny, nx),
    transform=transform_sub,
    fill=0,
    dtype=np.uint8,
    all_touched=False
)

plt.figure(figsize=(6,5))
plt.imshow(mask_tot_ice, cmap="magma", origin="upper",
           extent=extent)
plt.colorbar(label="True=1")
plt.title("Mask of all ice covered area in the sub extent")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.show()




for i in range(Zi.shape[0]):
    for j in range(Zi.shape[1]):
        point = Point(Xi[i, j], Yi[i, j])
        if not mask_geom.contains(point):
            mask[i, j] = True

Zi_masked = np.ma.array(Zi, mask=mask)
Surfi_masked = np.ma.array(Surfi, mask=mask)

plt.scatter(Xi, Yi,c=Zi_masked, s=3)
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="CTS depth")
plt.show()


if biharmonic :
    bed = z.copy()
    mask = np.isnan(bed) & mask_tot_ice

    # replace NaN temporarily (skimage requires finite values)
    bed_temp = bed.copy()
    bed_temp[mask] = 0

    bed_filled = inpaint.inpaint_biharmonic(bed_temp, mask)






plt.figure(figsize=(8, 6))

plt.pcolormesh(
    Xi, Yi, Zi_masked,
    shading="auto", cmap="terrain" 
)

plt.colorbar(label="Profondeur")
plt.xlabel("X")
plt.ylabel("Y")


plt.figure(figsize=(8, 6))

plt.pcolormesh(
    Xi, Yi, Surfi_masked,
    shading="auto", cmap="terrain"
)

plt.colorbar(label="Surf")
plt.xlabel("X")
plt.ylabel("Y")


# Optionnel : afficher le contour du shapefile
gdf_one.boundary.plot(ax=plt.gca(), color="black", linewidth=1)

#plt.gca().invert_yaxis()  # utile en bathymétrie
plt.show()










#dx = xi[1] - xi[0]   # ou la valeur que tu as fixée
#dy = yi[1] - yi[0]

#x_min = xi.min()
#y_max = yi.max()

#transform = from_origin(x_min,y_max,dx,dy)



# Convertir en array classique
#Z = np.array(Zi_masked)

# Valeur NoData
nodata = -9999

# Remplacer les NaN / masques
#Z_filled = np.where(np.isnan(Zi_masked), nodata, Zi_masked)

# Important : inversion verticale (matrice → coordonnées)
Z_filled = np.flipud(Zi_masked)
Surf_filled = np.flipud(Surfi_masked)

thk = Surf_filled - Z_filled



plt.figure(figsize=(8, 6))

plt.pcolormesh(
    Xi, Yi, thk,
    shading="auto", cmap="Blues"
)

plt.colorbar(label="thk")
plt.xlabel("X")
plt.ylabel("Y")


# Optionnel : afficher le contour du shapefile
gdf_one.boundary.plot(ax=plt.gca(), color="black", linewidth=1)

#plt.gca().invert_yaxis()  # utile en bathymétrie
plt.show()


TIF = False
if TIF : 

    output_tif = "profondeur_interpolee.tif"
    epsg_code = 32633  # je pense ?!

    with rasterio.open(
        output_tif,
        "w",
        driver="GTiff",
        height=Z_filled.shape[0],
        width=Z_filled.shape[1],
        count=1,
        dtype=Z_filled.dtype,
        crs=f"EPSG:{epsg_code}",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(Z_filled, 1)




ds_out = xr.Dataset(
    {
        "topg": (("y","x"), Z_filled),
        "usurf": (("y","x"),Surf_filled),
        "thk": (("y","x"), thk),
        "icemask": (("y","x"), mask_tot_ice)
    },
    coords={
        "x": (("x",), xi),
        "y": (("y",), yi)
    }
)

ds_out["topg"].attrs = {
    "long_name": "Bedrock topography",
    "units": "m",
    "description": "bedrock interpolated",
    "_FillValue": np.nan
}

ds_out["thk"].attrs = {
    "long_name": "Thickness",
    "units": "m",
    "description": "bedrock interpolated",
    "_FillValue": np.nan
}

ds_out["usurf"].attrs = {
    "long_name": "Surface elevation",
    "units": "m",
    "description": "bedrock interpolated",
    "_FillValue": np.nan
}




ds_out["icemask"].attrs = {
    "long_name": "Ice mask 1936",
    "units": "1 = ice, 0 = no ice"
}


ds_out.attrs = {
    "title": "Input bed",
    "projection": "EPSG:32633",
    "description": "Generated from Mannerfet et al. GPRs 2023i, res {dx}m"
}

dir_output = "./BEDS_FROM_GPR"
output_path = os.path.join(dir_output, f"Input_{glacier_name}_only_GPR_res_{dx}m.nc")
ds_out.to_netcdf(output_path)
print("NetCDF saved at:", output_path)


