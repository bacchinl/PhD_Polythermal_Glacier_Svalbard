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

year_surf = 2024
nom_glacier = "Edvardbreen" #"Ragna-Mariebreen" ou "Mettebreen"

if nom_glacier == "Ragna-Mariebreen" :
##### sveigbreen
    x_min, x_max, y_min, y_max= [552360, 558800, 8631600, 8640850] #please, epsg 32633

if nom_glacier == "Mettebreen" :
##### elfen
    x_min, x_max, y_min, y_max= [549900, 555750, 8635600, 8642200]

elif nom_glacier == "Kroppbreen":
    x_min, x_max, y_min, y_max= [553300, 559700, 8644100, 8654200]
elif nom_glacier == "Vallåkrabreen":
    x_min, x_max, y_min, y_max= [546100, 554000, 8639300, 8648700]

elif nom_glacier == "Edvardbreen":
    x_min, x_max, y_min, y_max= [553400, 566400, 8639700, 8655200]





src_bed ="GPR"


poly_subextent = Polygon([
    (x_min, y_min),
    (x_min, y_max),
    (x_max, y_max),
    (x_max, y_min)
])
##################### DOWNLOAD DATAS ########
###### NC FURST #########



data_path = "."

ds_bed = xr.open_dataset(os.path.join(data_path, f"Bedrock_{nom_glacier}_{src_bed}_res_20m.nc"))

print(ds_bed.attrs)

bed = ds_bed['topg']
print("--- bed shape : ", bed.shape)

x_bed = ds_bed['x'].values
y_bed = ds_bed['y'].values
print("X bed: ", np.min(x_bed),  np.max(x_bed))


#thk_sub = thk.sel(x=slice(x_min, x_max),y=slice(y_min, y_max))
#bed_sub = bed.sel(x=slice(x_min, x_max),y=slice(y_max, y_min))

if y_bed[0] < y_bed[-1]:  # y croissant
    bed_sub = bed.sel(x=slice(x_min, x_max),
                      y=slice(y_min, y_max))
else:  # y décroissant
    bed_sub = bed.sel(x=slice(x_min, x_max),
                      y=slice(y_max, y_min))



x_bed_sub = bed_sub['x'].values
y_bed_sub = bed_sub['y'].values
bed_array = bed_sub.values
#bed_array = np.flipud(bed_array)

print(len(y_bed_sub))
print("--- bed sub shape : ", bed_sub.shape)


#bed_array = np.flipud(bed_array)
plt.figure(figsize=(12,10))
plt.imshow(bed_array, cmap="terrain", origin="upper")
plt.show()
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
print("bed y first/last:", y_bed_sub[0], y_bed_sub[-1])

####### TIF DEM 1936 & 1990 #########



if year_surf == 1936:
    path_dems = "../../PhD_Lucie/DATA/DEMS/DEM-1936-NPI/"
    tif_path = os.path.join(path_dems, "dem_1936_all_Svalbard_50m_v3.tif")
elif year_surf == 1990 :
    path_dems = "../../PhD_Lucie/DATA/DEMS/DEM-1990-NPI/NP_S0_DTM5_2011_25163_33"
    tif_path = os.path.join(path_dems, "S0_DTM5_2011_25163_33.tif")
elif year_surf == 2024 :
    path_dems = "../../../PhD_Lucie/DATA/DEMS/DEM-2024-EM-Ragna-Mette/"
    tif_path = os.path.join(path_dems, "edvard_mette_ragna_kropp_dem_2024.tif")


with rasterio.open(tif_path) as src:
    usurf = src.read(1).astype(float)     # lecture de la couche 1
    nodata = src.nodata
    transform = src.transform
    bounds = src.bounds
    res = src.res[0]                      # résolution (m)
    nx, ny = src.width, src.height
    crs = src.crs



print("---SURFACE ELEVATION---")
print("      CRS :", crs)
print("      Bounds :", bounds)
print("      Resolution :", res)
print("      Shape :", usurf.shape)

if nodata is not None:
    usurf[usurf == nodata] = np.nan




print("Min/max after nan :", np.nanmin(usurf), np.nanmax(usurf))

x_min_tif, y_min_tif, x_max_tif, y_max_tif = bounds

# x augmente vers la droite
x = np.arange(nx) * res + x_min_tif

# y diminue à mesure qu'on descend les lignes
y = y_max_tif - np.arange(ny) * res
ix = np.where((x >= x_min) & (x <= x_max))[0]
iy = np.where((y >= y_min) & (y <= y_max))[0]

print("ix range :", ix[0], ix[-1])
print("iy range :", iy[0], iy[-1])

usurf_sub = usurf[np.ix_(iy, ix)]

print("--- usurf_sub shape :", usurf_sub.shape)
print("    usurf_sub min/max :", np.nanmin(usurf_sub), np.nanmax(usurf_sub))

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

path_shp_2010 = "~/PhD_Lucie/DATA/GLACIER_OUTLINES/CryoClim_GAO_SJ_2001-2010/"
shp_path_2010 = os.path.join(path_shp_2010, "CryoClim_GAO_SJ_2001-2010.shp")


if year_surf == 1936 :
    gdf = gpd.read_file(shp_path_1936)
elif year_surf == 1990 :
    gdf = gpd.read_file(shp_path_1990) 
elif year_surf > 2010 :
    gdf = gpd.read_file(shp_path_2010) 

#print(gdf.crs)
#print(gdf.head())

################# PROJECT EVERYTHING ON USURF ########

# Dimensions du subset
ny, nx = usurf_sub.shape

# Transform correspondant
transform_sub = from_bounds(x_min, y_min, x_max, y_max, nx, ny)

gdf_one = gdf[gdf["NAME"] == nom_glacier]

#if nom_glacier == "Vallåkrabreen":
 #   gdf_one = gdf[gdf["IDENT"] == 13413.10000]


print("GDF ", nom_glacier,  gdf_one)

shapes = [(geom, 1) for geom in gdf_one.geometry]

mask_main = rasterize(
    shapes=shapes,
    out_shape=(ny, nx),
    transform=transform_sub,
    fill=0,          # hors glacier
    dtype=np.uint8,
    all_touched=False  # True = étend un peu le glacier, False = strict
)


print(f"---mask_main shape (only {nom_glacier}):", mask_main.shape)
print("   Nb pixels glacier:", np.sum(mask_main))

#if nom_glacier == "Sveigbreen" :
#    glaciers = ["Sveigbreen", "Jinnbreen", "Skruisbreen"]#, "S°atebreen"] # otherwise, it is not pretty, we need the bedrock for all of them

gdf_multi = gdf[gdf.geometry.intersects(poly_subextent)] 

#gdf_multi = gdf[gdf["NAME"].isin(glaciers)]
shapes = [(geom, 1) for geom in gdf_multi.geometry]

mask_tot_ice = rasterize(
    shapes=shapes,
    out_shape=(ny, nx),
    transform=transform_sub,
    fill=0,
    dtype=np.uint8,
    all_touched=False
)


####### BED REPROJ


# tableau de sortie avec la même taille que usurf_sub
bed_reproj = np.full_like(usurf_sub, np.nan, dtype=float)

if year_surf < 2010 :
    transform_bed = rasterio.transform.Affine(
        dx_bed_sub, 0, x_bed_sub.min(),
        0, -dy_bed_sub, y_bed_sub.max()
    )


elif year_surf > 2010 :
    transform_bed = rasterio.transform.Affine(
        dx_bed_sub, 0, x_bed_sub.min()- dx_bed_sub / 2,
        0, -dy_bed_sub, y_bed_sub.max() + dy_bed_sub / 2
    )   

reproject(
    source=bed_array,
    destination=bed_reproj,
    src_transform=transform_bed,
    src_crs="EPSG:32633",
    dst_transform=transform_sub,
    dst_crs="EPSG:32633",
    resampling=Resampling.bilinear
)

print("---bed_reproj shape:", bed_reproj.shape)
print("   bed_reproj min/max :", np.nanmin(bed_reproj), np.nanmax(bed_reproj))


plt.figure(figsize=(6,5))
plt.imshow(mask_tot_ice, cmap="magma", origin="upper",
           extent=[x[ix[0]], x[ix[-1]], y[iy[-1]], y[iy[0]]])
plt.colorbar(label="True=1")
plt.title("Mask of all ice covered area in the sub extent")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.show()

usurf_clean = np.nan_to_num(usurf_sub, nan=0)
bed_reproj = np.where(bed_reproj>usurf_sub, usurf_sub, bed_reproj)

plt.figure(figsize=(12,10))
plt.imshow(bed_reproj, cmap="terrain", origin="upper",
           extent=[x[ix[0]], x[ix[-1]], y[iy[-1]], y[iy[0]]],vmin = 20, vmax=600)
plt.colorbar(label="Altitude (m)")
plt.title("Bedrock")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
#plt.show()

plt.figure(figsize=(12,10))
plt.imshow(usurf_sub, cmap="terrain", origin="upper",
           extent=[x[ix[0]], x[ix[-1]], y[iy[-1]], y[iy[0]]],vmin = 20, vmax=600)
plt.colorbar(label="Altitude (m)")
plt.title("Surface")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.show()


plt.figure(figsize=(10,8))
plt.imshow(usurf_sub - bed_reproj, cmap="RdBu", vmin=-200, vmax=200)
plt.colorbar(label="surface - bed")
plt.title("Difference")
plt.show()
print("surface min/max:", np.nanmin(usurf_sub), np.nanmax(usurf_sub))
print("bed min/max:", np.nanmin(bed_reproj), np.nanmax(bed_reproj))
print("NB PIXEL BED > SURF",np.sum(bed_reproj > usurf_sub)
        )

print(usurf_sub.shape, bed_reproj.shape)
print("dx surface:", transform_sub.a)
print("dy surface:", transform_sub.e)

print("dx bed:", transform_bed.a)
print("dy bed:", transform_bed.e)

#bed_reproj = np.where(bed_reproj>usurf_sub, usurf_sub, bed_reproj)



thk = usurf_sub-bed_reproj
thk_glacier=thk
thk_glacier = np.where(mask_main == 1, thk, 0)

thk_sin_neg = np.where(thk>0, thk, 0.1)
#thk_sin_neg = np.where(mask_tot_ice == 1, thk_sin_neg, 0)


#usurf_masked =   np.where(mask_main == 1, usurf_sub, np.nan)
#bed_masked =   np.where(mask_main == 1, bed_reproj, np.nan)


#bed_reproj = np.where(bed_reproj>usurf_sub, usurf_sub, bed_reproj)


plt.figure(figsize=(12,10))
plt.imshow(thk_sin_neg, cmap="Blues", origin="upper",
           extent=[x[ix[0]], x[ix[-1]], y[iy[-1]], y[iy[0]]], vmin = 0, vmax=200)
plt.colorbar(label="Altitude (m)")
plt.title(f"Thickness {year_surf}, after interpolation")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.show()

print("min max thk glacier ", np.nanmin(thk_glacier), np.nanmax(thk_glacier))

#print('x et y : ', x_sub, y_sub)



print("Thickness statistics:")
print("Min:", np.nanmin(thk))
print("Max:", np.nanmax(thk))
print("Mean:", np.nanmean(thk))
print("Non-zero pixels:", np.sum(thk_sin_neg > 0))
print("NaNs in thickness:", np.isnan(thk_sin_neg).sum())

print("NaNs in usurf", np.isnan(usurf_clean).sum())
print("NaNs in bed:", np.isnan(bed_reproj).sum())


######## Save ######




ds_out = xr.Dataset(
    {
        "topg": (("y","x"), bed_reproj),
        "usurf": (("y","x"), usurf_clean),
        "thk": (("y","x"), thk_sin_neg),
        "icemask": (("y","x"), mask_main)
    },
    coords={
        "x": (("x",), x_sub),   
        "y": (("y",), y_sub)    
    }
)

ds_out["topg"].attrs = {
    "long_name": "Bedrock topography",
    "units": "m",
    "description": "bedrock interpolated",
    "_FillValue": np.nan
}

ds_out["usurf"].attrs = {
    "long_name": "Surface elevation",
    "units": "m",
    "description": f"DEM {year_surf}",
    "_FillValue": np.nan
}

ds_out["thk"].attrs = {
    "long_name": f"Thickness",
    "units": "m",
    "description": "usurf - topg",
    "_FillValue": np.nan
}



ds_out["icemask"].attrs = {
    "long_name": f"Ice mask {year_surf}",
    "units": "1 = ice, 0 = no ice"
}


ds_out.attrs = {
    "title": f"Input dataset {nom_glacier} {year_surf}",
    "projection": "EPSG:32633",
    "description": f"Generated from homemade bedrock ({src_bed}), North polar institute surface DEM et glacier outline for {year_surf}"
}

dir_output = "./INPUTS"
if nom_glacier == "Ragna-Mariebreen":
    dir_output = "../../igm/igm/ragna_basic/data"
if nom_glacier == "Mettebreen":
    dir_output = "../../igm/igm/mette_basic/data"
if nom_glacier == "Kroppbreen":
    dir_output = "../../../igm/igm/kropp_basic/data"

output_path = os.path.join(dir_output, f"input_{nom_glacier}_{year_surf}_{src_bed}_bed.nc")
ds_out.to_netcdf(output_path)
print("NetCDF saved at:", output_path)

################## PLOTS ##########



plt.figure(figsize=(8,6))
plt.imshow(usurf_sub, cmap="terrain", origin='upper', vmin = 0,extent=[
        bed_sub.x.min(), bed_sub.x.max(),
        bed_sub.y.min(), bed_sub.y.max()
    ])
plt.colorbar(label=" Bedrock elevation (m)")
plt.title(f"Altitude m")
plt.xlabel("x ")
plt.ylabel("y ")
#plt.show()




plt.figure(figsize=(8,6))
plt.imshow(bed_sub, cmap="terrain", origin='lower', vmin = 0,extent=[
        bed_sub.x.min(), bed_sub.x.max(),
        bed_sub.y.min(), bed_sub.y.max()
    ])
plt.colorbar(label=" Bedrock elevation (m)")
plt.title(f"Altitude m")
plt.xlabel("x ")
plt.ylabel("y ")
#plt.show()




