import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator, splprep, splev
import imageio
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.interpolate import griddata
from skimage.restoration import inpaint

import plotly.graph_objects as go


nom_glacier = "Edvardbreen" #"Ragna-Mariebreen" ou "Mettebreen"
glacier_name = "edvardbreen"

#please, epsg 32633
if nom_glacier == "Mettebreen" :
    x_min, x_max, y_min, y_max= [549900, 558800, 8631600, 8642200] #please, epsg 32633
elif nom_glacier == "Kroppbreen":
    x_min, x_max, y_min, y_max= [553300, 559700, 8644100, 8654200]
elif glacier_name == "vallakrabreen":
    x_min, x_max, y_min, y_max= [546100, 554000, 8639300, 8648700]

elif glacier_name == "edvardbreen":
    x_min, x_max, y_min, y_max= [553400, 566400, 8639700, 8655200]

poly_subextent = Polygon([
    (x_min, y_min),
    (x_min, y_max),
    (x_max, y_max),
    (x_max, y_min)
])

source_bed = "GPR"

res = 20
##################### DOWNLOAD DATAS ########
###### NC FURST #########

if source_bed == "Furst":

    data_path = "~/PhD_Lucie/DATA/BEDROCK/Furst-svift-topo/"

    ds_thk = xr.open_dataset(os.path.join(data_path, "svift_v1_thickness.nc"))
    ds_bed = xr.open_dataset(os.path.join(data_path, "svift_v1_bed.nc"))

    print(ds_bed.attrs)

    thk = ds_thk['thi']
    bed = ds_bed['bed']

    x_bed = ds_bed['x'].values
    y_bed = ds_bed['y'].values



    #thk_sub = thk.sel(x=slice(x_min, x_max),y=slice(y_min, y_max))
    bed_sub = bed.sel(x=slice(x_min, x_max),y=slice(y_min, y_max))

    x_bed_sub = bed_sub['x'].values
    y_bed_sub = bed_sub['y'].values
    bed_array = bed_sub.values

    bed_array = np.flipud(bed_array)
    y_bed_sub = y_bed_sub[::-1]

    dx_bed_sub = x_bed_sub[1] - x_bed_sub[0]              # ~170 m attendu
    dy_bed_sub = y_bed_sub[1] - y_bed_sub[0]
    # coin SUPÉRIEUR GAUCHE
    x0_bed_sub = x_bed_sub.min()
    y0_bed_sub = y_bed_sub.max()

    print("--- bed sub shape : ", bed_sub.shape)
    print("    bed_sub min/max :", np.nanmin(bed_sub), np.nanmax(bed_sub))
    print("    bed_sub dx,dy :", dx_bed_sub, dy_bed_sub)
    print("    ",x_bed_sub[:5], x_bed_sub[-5:])
    print("    ",y_bed_sub[:5], y_bed_sub[-5:])

if source_bed == "GPR":
    data_path = "./BEDS_FROM_GPR"
    ds_bed = xr.open_dataset(os.path.join(data_path, f"Input_{glacier_name}_only_GPR_res_{res}m.nc"))

    bed = ds_bed['topg']

    x_bed = ds_bed['x'].values
    y_bed = ds_bed['y'].values
    bed_sub = bed.sel(x=slice(x_min, x_max),y=slice(y_min, y_max))

    x_bed_sub = bed_sub['x'].values
    y_bed_sub = bed_sub['y'].values
    bed_array = bed_sub.values

    #bed_array = np.flipud(bed_array)
    y_bed_sub = y_bed_sub[::-1]

    dx_bed_sub = x_bed_sub[1] - x_bed_sub[0]              # ~170 m attendu
    dy_bed_sub = y_bed_sub[1] - y_bed_sub[0]
    # coin SUPÉRIEUR GAUCHE
    x0_bed_sub = x_bed_sub.min()
    y0_bed_sub = y_bed_sub.max()

    print("--- bed sub shape : ", bed_sub.shape)
    print("    bed_sub min/max :", np.nanmin(bed_sub), np.nanmax(bed_sub))
    print("    bed_sub dx,dy :", dx_bed_sub, dy_bed_sub)
    print("    ",x_bed_sub[:5], x_bed_sub[-5:])
    print("    ",y_bed_sub[:5], y_bed_sub[-5:])
    

    plt.figure(figsize=(12,10))
    plt.imshow(bed_array, cmap="terrain", origin="upper",vmin = 20, vmax=600)
    plt.colorbar(label="Altitude (m)")
    plt.title("Bedrock GPR")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.show()





x_out = x_bed_sub #np.arange(x_min, x_max + res, res)
y_out = y_bed_sub #np.arange(y_min, y_max + res, res)

nx_out = len(x_out)
ny_out = len(y_out)


####### TIF DEM 1936 & 1990 #########


path_tifs = f"../../../PhD_Lucie/DATA/BEDROCK/Van_pelt"
tif_path_vp = os.path.join(path_tifs, "Bed_map.tif")



with rasterio.open(tif_path_vp) as src:
    bed_vp = src.read(1).astype(float)     # lecture de la couche 1
    nodata = src.nodata
    transform = src.transform
    bounds = src.bounds
    res_tif = src.res[0]                      # résolution (m)
    nx, ny = src.width, src.height
    crs = src.crs



print("-----------VAN PELT----------")
print("CRS :", crs)
print("Bounds :", bounds)
print("Resolution :", res_tif)
print("Shape :", bed_vp.shape)

if nodata is not None:
    bed_vp[bed_vp == nodata] = np.nan


print("Min/max after nan :", np.nanmin(bed_vp), np.nanmax(bed_vp))

x_min_tif, y_min_tif, x_max_tif, y_max_tif = bounds

# x augmente vers la droite
x = np.arange(nx) * res_tif + x_min_tif

# y diminue à mesure qu'on descend les lignes
y = y_max_tif - np.arange(ny) * res_tif
ix = np.where((x >= x_min) & (x <= x_max))[0]
iy = np.where((y >= y_min) & (y <= y_max))[0]

print("ix range :", ix[0], ix[-1])
print("iy range :", iy[0], iy[-1])

bed_vp_sub = bed_vp[np.ix_(iy, ix)]

print("--- usurf_sub shape :", bed_vp_sub.shape)
print("    usurf_sub min/max :", np.nanmin(bed_vp_sub), np.nanmax(bed_vp_sub))

x_sub = x[ix]
y_sub = y[iy]

print("x_sub :", x_sub[:5], "...", x_sub[-5:])
print("y_sub :", y_sub[:5], "...", y_sub[-5:])




#######################  IMPORTER ICE MASK 1990 #####

## reproject bed surf
path_shp_1936 = "~/PhD_Lucie/DATA/GLACIER_OUTLINES/CryoClim_GAO_SJ_1936-1972/"
shp_path_1936 = os.path.join(path_shp_1936, "CryoClim_GAO_SJ_1936-1972.shp")

path_shp_1990 = "~/PhD_Lucie/DATA/GLACIER_OUTLINES/CryoClim_GAO_SJ_1990/"
shp_path_1990 = os.path.join(path_shp_1990, "CryoClim_GAO_SJ_1990.shp")



gdf = gpd.read_file(shp_path_1936)
#print(gdf.crs)
#print(gdf.head())

################# PROJECT EVERYTHING ON USURF ########

# Dimensions du subset
ny_vp, nx_vp = bed_vp_sub.shape

# Transform correspondant
transform_sub = from_bounds(x_min, y_min, x_max, y_max, nx_out, ny_out)

bed_vp_reproj = np.full(
    (ny_out, nx_out),
    np.nan,
    dtype=float
)

#if nom_glacier == "Sveigbreen" :
#    glaciers = ["Sveigbreen", "Jinnbreen", "Skruisbreen"]#, "S°atebreen"] # otherwise, it is not pretty, we need the bedrock for all of them

gdf_multi = gdf[gdf.geometry.intersects(poly_subextent)] 

#gdf_multi = gdf[gdf["NAME"].isin(glaciers)]
shapes = [(geom, 1) for geom in gdf_multi.geometry]

mask_tot_ice = rasterize(
    shapes=shapes,
    out_shape=(ny_out, nx_out),
    transform=transform_sub,
    fill=0,
    dtype=np.uint8,
    all_touched=False
)


print("Shape_mask ", mask_tot_ice.shape)

####### BED REPROJ


# tableau de sortie avec la même taille que usurf_sub
bed_vp_reproj = np.full(
    (ny_out, nx_out),
    np.nan,
    dtype=float
)


reproject(
    source=bed_vp,
    destination=bed_vp_reproj,
    src_transform=transform,
    src_crs=crs,
    dst_transform=transform_sub,
    dst_crs="EPSG:32633",
    resampling=Resampling.bilinear
)

print("---bed_reproj shape:", bed_vp_reproj.shape)
print("   bed_reproj min/max :", np.nanmin(bed_vp_reproj), np.nanmax(bed_vp_reproj))

print("Final resolution:")
print(
    (x_out[1]-x_out[0]),
    (y_out[1]-y_out[0])
)
################################# CUT AND INTERPOLATION #################

bed_vp_cut = np.where(mask_tot_ice == 1, np.nan, bed_vp_reproj)

bed_gpr_cut = np.where(mask_tot_ice == 0, np.nan, bed_array)

bed_tot = np.where(np.isnan(bed_vp_cut), bed_gpr_cut, bed_vp_cut)

plt.figure(figsize=(6,5))
plt.imshow(mask_tot_ice, cmap="magma", origin="upper",
           extent=[x[ix[0]], x[ix[-1]], y[iy[-1]], y[iy[0]]])
plt.colorbar(label="True=1")
plt.title("Mask of all ice covered area in the sub extent")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.show()


plt.figure(figsize=(12,10))
plt.imshow(bed_tot, cmap="terrain", origin="upper",
           extent=[x[ix[0]], x[ix[-1]], y[iy[-1]], y[iy[0]]],vmin = 20, vmax=600)
plt.colorbar(label="Altitude (m)")
plt.title("Bedrock 1990, before interpolation")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.show()


def fill_bed_nan_biharmonic(bed_tot):
    """
    Fill NaN values in a bedrock array using biharmonic inpainting.

    Parameters
    ----------
    bed_tot : 2D numpy array
        Bedrock elevation with NaNs where data is missing.

    Returns
    -------
    bed_filled : 2D numpy array
        Bedrock with NaNs filled.
    """

    bed = bed_tot.copy()

    # mask = True where values must be filled
    mask = np.isnan(bed)

    # replace NaN temporarily (skimage requires finite values)
    bed_temp = bed.copy()
    bed_temp[mask] = 0

    # biharmonic inpainting
    bed_filled = inpaint.inpaint_biharmonic(bed_temp, mask)

    return bed_filled


def fill_nans_griddata(arr):
    ny, nx = arr.shape

    # grille
    yy, xx = np.indices(arr.shape)

    # points valides
    mask = ~np.isnan(arr)
    points = np.column_stack((xx[mask], yy[mask]))
    values = arr[mask]

    # points à remplir
    points_nan = np.column_stack((xx[~mask], yy[~mask]))

    # interpolation linéaire puis nearest si trou en bord
    filled = arr.copy()
    filled[~mask] = griddata(points, values, points_nan, method="linear")

    # fallback nearest pour les zones non interpolables
    still_nan = np.isnan(filled)
    if np.any(still_nan):
        filled[still_nan] = griddata(points, values,
                                     np.column_stack((xx[still_nan], yy[still_nan])),
                                     method="nearest")

    return filled

print( "-------- Interpolation ---------")
#bed_tot_filled = fill_nans_griddata(bed_tot)

bed_tot_filled = fill_bed_nan_biharmonic(bed_tot)
plt.figure(figsize=(12,10))
plt.imshow(bed_tot_filled, cmap="terrain", origin="upper",
           extent=[x[ix[0]], x[ix[-1]], y[iy[-1]], y[iy[0]]], vmin = 20, vmax=600)
plt.colorbar(label="Altitude (m)")
plt.title("Bedrock 1990, after interpolation")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.show()




X, Y = np.meshgrid(x_out, y_out)


fig = go.Figure()
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')


thk = ax.plot_surface(
    X,
    Y,
    bed_tot_filled,
    cmap='terrain',
    linewidth=0,
    antialiased=False
)


fig.colorbar(thk, shrink=0.6, label='Surface elevation (m)')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Elevation (m)')

plt.show()


######## Save ######




ds_out = xr.Dataset(
    {
        "topg": (("y","x"), bed_tot_filled),
        "icemask": (("y","x"), mask_tot_ice)
    },
    coords={
        "x": (("x",), x_out),   
        "y": (("y",), y_out)    
    }
)

ds_out["topg"].attrs = {
    "long_name": "Bedrock topography",
    "units": "m",
    "description": "bedrock interpolated",
    "_FillValue": 0    
}


ds_out["icemask"].attrs = {
    "long_name": "Ice mask 1936",
    "units": "1 = ice, 0 = no ice"
}


ds_out.attrs = {
    "title": "Input dataset Sveigbreen",
    "projection": "EPSG:32633",
    "description": f"Generated from Svift from J. Furst and van pelt bedrock, using biharmonic methods to fill 1936 ice mask from NPI, res {res}m"
}

dir_output = "."
output_path = os.path.join(dir_output, f"Bedrock_{nom_glacier}_{source_bed}_res_{res}m.nc")
ds_out.to_netcdf(output_path)
print("NetCDF saved at:", output_path)

