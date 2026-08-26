
import os
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio

from shapely.geometry import Polygon
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling

# =========================================================
# PARAMETERS
# =========================================================

glacier_name = "kroppbreen"
glacier_name_shp = "Kroppbreen"

x_min, x_max = 553300, 559700
y_min, y_max = 8644100, 8654200

res = 10  # FINAL RESOLUTION (IMPORTANT)

epsg = 32633

# =========================================================
# GRID 10 m (UNIQUE GRID)
# =========================================================

x_out = np.arange(x_min, x_max + res, res)
y_out = np.arange(y_min, y_max + res, res)

nx = len(x_out)
ny = len(y_out)

transform = from_bounds(
    x_min, y_min,
    x_max, y_max,
    nx, ny
)

# =========================================================
# LOAD VAN PELT BED
# =========================================================

tif_path = "../../../PhD_Lucie/DATA/BEDROCK/Van_pelt/Bed_map.tif"

with rasterio.open(tif_path) as src:
    bed_vp = src.read(1).astype(float)
    nodata = src.nodata
    src_transform = src.transform
    src_crs = src.crs

if nodata is not None:
    bed_vp[bed_vp == nodata] = np.nan

# =========================================================
# REPROJECT VAN PELT TO 10 m GRID
# =========================================================

bed_vp_10m = np.full((ny, nx), np.nan, dtype=float)

reproject(
    source=bed_vp,
    destination=bed_vp_10m,
    src_transform=src_transform,
    src_crs=src_crs,
    dst_transform=transform,
    dst_crs=f"EPSG:{epsg}",
    resampling=Resampling.bilinear
)

# =========================================================
# LOAD GLACIER OUTLINE
# =========================================================

shp_path = "~/PhD_Lucie/DATA/GLACIER_OUTLINES/CryoClim_GAO_SJ_1990/CryoClim_GAO_SJ_1990.shp"

gdf = gpd.read_file(shp_path)

gdf = gdf[gdf["NAME"] == glacier_name_shp]

if len(gdf) == 0:
    raise ValueError("Glacier name not found in shapefile")

glacier_geom = gdf.unary_union

# =========================================================
# ICE MASK (10 m GRID)
# =========================================================

mask_ice = rasterize(
    [(glacier_geom, 1)],
    out_shape=(ny, nx),
    transform=transform,
    fill=0,
    dtype=np.uint8
)

# =========================================================
# MASK BEDROCK OUTSIDE GLACIER
# =========================================================

bed_vp_10m[mask_ice == 0] = np.nan

# =========================================================
# OPTIONAL: CLEAN SMALL ARTIFACTS
# =========================================================

# (important for IGM stability)
bed_vp_10m = np.where(bed_vp_10m < 0, np.nan, bed_vp_10m)

# =========================================================
# FINAL COORD ADJUSTMENT
# =========================================================

y_out_flip = y_out[::-1]
bed_final = np.flipud(bed_vp_10m)
mask_final = np.flipud(mask_ice)

# =========================================================
# EXPORT NETCDF (IGM READY)
# =========================================================

ds = xr.Dataset(
    {
        "topg": (("y", "x"), bed_final),
        "icemask": (("y", "x"), mask_final)
    },
    coords={
        "x": x_out,
        "y": y_out_flip
    }
)

ds["topg"].attrs = {
    "long_name": "Bedrock elevation",
    "units": "m",
    "description": "Van Pelt bed reprojected to 10 m grid for IGM"
}

ds["icemask"].attrs = {
    "long_name": "Ice mask",
    "units": "binary"
}

ds.attrs = {
    "title": f"{glacier_name} bed for IGM",
    "resolution_m": res,
    "projection": f"EPSG:{epsg}"
}

output = f"IGM_bed_{glacier_name}_{res}m.nc"
ds.to_netcdf(output)

print("Saved:", output)
print("Final resolution:", res, "m")


