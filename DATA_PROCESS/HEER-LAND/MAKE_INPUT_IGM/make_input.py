import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator, splprep, splev, NearestNDInterpolator
import imageio
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.interpolate import griddata

from utils_make_input import * 


year_surf = 1990
nom_glacier = "Global" #"Ragna-Mariebreen" ou "Mettebreen"

if nom_glacier == "Global" :
    x_min, x_max, y_min, y_max= [543500, 566500, 8633000, 8655000]

glaciers_to_exclude = ["Nordsysselbreen", "Bakaninbreen", "Inglefieldbreen", "Sørbullbreen", "Schleelebreen", "Helsingborgbreen", "Peisbreen", "Andrinebreen"] ## I want the littles snowpatchs but no big tide water glaciers in this simu and border ones

extent = [x_min, x_max, y_min, y_max]

src_bed ="GPR_and_vp"


poly_subextent = Polygon([
    (x_min, y_min),
    (x_min, y_max),
    (x_max, y_max),
    (x_max, y_min)
])

res_target = 50
##################### DOWNLOAD DATAS ########
###### NC FURST #########



data_path = "BEDS_PROCESSED"

ds_bed = xr.open_dataset(os.path.join(data_path, f"Bedrock_{nom_glacier}_{src_bed}_res_50m.nc"))

bed = ds_bed['topg']

BED = project_tar_grid( extent,
    bed, res_target,
    method="linear",
)

####### TIF SURFACE DEM 1936, 1990, 2010, 2024 #########



if year_surf == 1936:
    path_dems = "../../../PhD_Lucie/DATA/DEMS/DEM-1936-NPI/"
    tif_path = os.path.join(path_dems, "dem_1936_all_Svalbard_50m_v3.tif")
elif year_surf == 1990 :
    path_dems = "../../../PhD_Lucie/DATA/DEMS/DEM-1990-NPI/NP_S0_DTM20_199095_33"
    tif_path = os.path.join(path_dems, "S0_DTM20_199095_33.tif")
elif year_surf == 2010 :
    path_dems = "../../../PhD_Lucie/DATA/DEMS/DEM-1990-NPI/NP_S0_DTM5_2011_25163_33"
    tif_path = os.path.join(path_dems, "S0_DTM5_2011_25163_33.tif")
elif year_surf == 2024 :
    path_dems = "../../../PhD_Lucie/DATA/DEMS/DEM-2024-EM-Ragna-Mette/"
    tif_path = os.path.join(path_dems, "edvard_mette_ragna_kropp_dem_2024.tif")




USURF, x_usurf, y_usurf = project_tif_to_grid(
    tif_path,
    extent,
    d_target=res_target,
)



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



################# PROJECT EVERYTHING ON USURF ########

# Dimensions du subset
ny, nx = USURF.shape

# Transform correspondant
transform_sub = from_bounds(x_min, y_min, x_max, y_max, nx, ny)


#if nom_glacier == "Sveigbreen" :
#    glaciers = ["Sveigbreen", "Jinnbreen", "Skruisbreen"]#, "S°atebreen"] # otherwise, it is not pretty, we need the bedrock for all of them



gdf_multi = gdf[gdf.geometry.intersects(poly_subextent)& (~gdf["NAME"].isin(glaciers_to_exclude))] 
#gdf_multi = gdf[gdf["NAME"].isin(glaciers)] ## or

shapes = [(geom, 1) for geom in gdf_multi.geometry]

mask_tot_ice = rasterize(
    shapes=shapes,
    out_shape=(ny, nx),
    transform=transform_sub,
    fill=0,
    dtype=np.uint8,
    all_touched=False
)

plot_var(mask_tot_ice, "Mask for ice", "magma",0,1)

#################################### ADDD CLIMATE #####################
path_clim = "../../CODES_TANCREDE_CLIMATE"

ds_MAT = xr.open_dataset(os.path.join(path_clim, f"MAT_CARRA_1990_2024_epsg32633_1km.nc"))
ds_MST = xr.open_dataset(os.path.join(path_clim, f"MST_CARRA_1990_2024_epsg32633_1km.nc"))
ds_PREC = xr.open_dataset(os.path.join(path_clim, f"PREC_CARRA_1990_2024_epsg32633_1km.nc"))



MAT = project_tar_grid( extent,
    ds_MAT["t2m_clim_32633"], res_target,
    method="linear",
)
MAT.data = fill_nan_nearest(MAT.data)

MST = project_tar_grid( extent,
    ds_MST["t2m_clim_32633"], res_target,
    method="linear", 
)
MST.data = fill_nan_nearest(MST.data)

PREC = project_tar_grid( extent,
    ds_PREC["total_precip_32633"], res_target,
    method="linear",
)
PREC.data = fill_nan_nearest(PREC.data)

print_var_clim(MAT, "MAT")
print_var_clim(MST, "MST")
print_var_clim(PREC, "Precips")

plot_var(MAT, "MAT", "Spectral_r", -10,-5)
plot_var(MST, "MST", "Spectral_r", 0,5)
plot_var(PREC, "Precipitation", "GnBu",100,400)
############## A bit of cleaning ##############




USURF_clean = np.nan_to_num(USURF, nan=0)
BED_clean = np.where(BED>USURF, USURF_clean, BED)

plot_var(USURF_clean, "usurf clean", "terrain",0,800)
plot_var(BED_clean, "bed clean", "terrain",0,800)



thk = USURF_clean-BED_clean

thk_glacier = np.where(mask_tot_ice == 1, thk, 0)

thk_sin_neg = np.where(thk_glacier>0, thk_glacier, 0.1)


plot_var(thk_sin_neg, "clean thickness", "Blues",0,400)

print("min max thk glacier ", np.nanmin(thk_glacier), np.nanmax(thk_glacier))


print("Thickness statistics:")
print("Min:", np.nanmin(thk))
print("Max:", np.nanmax(thk))
print("Mean:", np.nanmean(thk))
print("Non-zero pixels:", np.sum(thk_sin_neg > 0))
print("NaNs in thickness:", np.isnan(thk_sin_neg).sum())

print("NaNs in usurf", np.isnan(USURF_clean).sum())
print("NaNs in bed:", np.isnan(BED_clean).sum())


x_target, y_target =make_grid_target(extent, res_target) 
print("X :",x_target[:5] )

print(x_target.dtype)
print(y_target.dtype)
######## Save ######




ds_out = xr.Dataset(
    {
        "usurf": (("y","x"), USURF_clean),
        "thk": (("y","x"), thk_sin_neg),
        "icemask": (("y","x"), mask_tot_ice),
        "air_temp": (("y","x"), MAT.data),
        "air_temp_summer": (("y","x"), MST.data),
        "precipitation": (("y","x"), PREC.data)
    },
    coords={
        "x": (("x",), x_target),   
        "y": (("y",), y_target)    
    }
)


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

ds_out["air_temp"].attrs = {
    "long_name": f"Mean Annual air temperature",
    "units": "C",
    "description": "From CARRA 1990-2024",
    "_FillValue": np.nan
}

ds_out["air_temp_summer"].attrs = {
    "long_name": f"Mean summer air temperature",
    "units": "°C",
    "description": "From CARRA 1990-2024",
    "_FillValue": np.nan
}

ds_out["precipitation"].attrs = {
    "long_name": f"Total precipitation per year (annual mean)",
    "units": "mm/year",
    "description": "From CARRA 1990-2024",
    "_FillValue": np.nan
}

ds_out["x"].attrs = {
    "standard_name": "projection_x_coordinate",
    "long_name": "x coordinate of projection",
    "units": "m",
}

ds_out["y"].attrs = {
    "standard_name": "projection_y_coordinate",
    "long_name": "y coordinate of projection",
    "units": "m",
}


ds_out.attrs = {
    "title": f"Input dataset {nom_glacier} {year_surf}",
    "projection": "EPSG:32633",
    "description": f"Generated from homemade bedrock ({src_bed}), North polar institute surface DEM et glacier outline for {year_surf}"
}

dir_output = "./INPUT"
if nom_glacier == "Ragna-Mariebreen":
    dir_output = "../../igm/igm/ragna_basic/data"
if nom_glacier == "Mettebreen":
    dir_output = "../../igm/igm/mette_basic/data"
if nom_glacier == "Kroppbreen":
    dir_output = "../../../igm/igm/kropp_basic/data"

output_path = os.path.join(dir_output, f"input_{nom_glacier}_{year_surf}_{src_bed}_bed.nc")
ds_out = ds_out.rio.write_crs(32633)
ds_out.to_netcdf(output_path)
print("NetCDF saved at:", output_path)


