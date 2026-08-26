
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator, splprep, splev, NearestNDInterpolator
import imageio
import rasterio
from rasterio.transform import from_bounds, from_origin
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.interpolate import griddata


##################### REPROJECTION THINGY  ######################


def project_tar_grid(extent, da, res, method="linear"):

    xmin, xmax, ymin, ymax = extent
    x_target = np.arange(xmin, xmax + res, res)
    y_target = np.arange(ymax, ymin -res , -res)

    
    if da.x[0] > da.x[-1]:
        da = da.sortby("x")

    # Adapter automatiquement le sens de y
    if (da.y[0] < da.y[-1]) != (y_target[0] < y_target[-1]):
        da = da.sortby("y", ascending=(y_target[0] < y_target[-1]))


    # Crop (adapter le sens de y)
    da = da.sel(
        x=slice(xmin, xmax),
        y=slice(ymax, ymin),
    )
    #plot_var(da, "clim before interp", "Spectral_r")

    return da.interp(
        x=x_target,
        y=y_target,
        method=method,
    )



def project_tif_to_grid(
    tif_path,
    extent,
    d_target,
    resampling=Resampling.bilinear,
):
    """
    Reproject a GeoTIFF onto the target grid.

    Parameters
    ----------
    tif_path : str
        Path to the GeoTIFF.
    extent : (xmin, xmax, ymin, ymax)
        Target extent.
    d_target : float
        Target resolution (m).
    resampling : rasterio.warp.Resampling
        Resampling method.

    Returns
    -------
    array : ndarray
        Reprojected raster.
    x : ndarray
    y : ndarray
    """

    xmin, xmax, ymin, ymax = extent

    x = np.arange(xmin, xmax + d_target, d_target)
    y = np.arange(ymax, ymin - d_target, -d_target)

    dst = np.full((len(y), len(x)), np.nan, dtype=np.float32)

    dst_transform = from_origin(
        xmin,
        ymax,
        d_target,
        d_target,
    )

    with rasterio.open(tif_path) as src:

        src_array = src.read(1).astype(np.float32)

        if src.nodata is not None:
            src_array[src_array == src.nodata] = np.nan

        reproject(
            source=src_array,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,
            resampling=resampling,
            dst_nodata=np.nan,
        )

    return dst, x, y



def make_grid_target(extent, res):
    xmin, xmax, ymin, ymax = extent
    x_target = np.arange(xmin, xmax + res, res, dtype=np.float32)
    y_target = np.arange(ymax, ymin -res , -res, dtype=np.float32)

    return x_target,y_target

########### These nans on the side are messing with igm #####

def fill_nan_nearest(arr):
    """
    Fill NaNs of a 2D array using nearest-neighbor interpolation.
    """

    arr = arr.copy()

    mask = np.isfinite(arr)

    yy, xx = np.indices(arr.shape)

    interp = NearestNDInterpolator(
        np.column_stack((xx[mask], yy[mask])),
        arr[mask],
    )

    arr[~mask] = interp(xx[~mask], yy[~mask])

    return arr
#################### PLOTING STUFF ####################

def print_var_clim(arr, name):
     print(
             f"{name} : finite = {np.isfinite(arr.values).sum()} / {arr.size}, "
        f"NaNs = {np.isnan(arr).sum()}, "
        f"min = {np.nanmin(arr):.3f}, "
        f"max = {np.nanmax(arr):.3f}",
    )


def plot_var(arr, name, cmap, vmin, vmax):     
     plt.figure(figsize=(10,8))
     plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
     plt.colorbar(label=f"{name}")
     plt.title(f"{name}")
     plt.show()



