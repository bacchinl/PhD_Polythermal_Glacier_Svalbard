import os
import numpy as np
import xarray as xr
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt

from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from shapely.geometry import Polygon


####### COMPARE AND SAVE BEDROCK AT THE RIGHT RESOLUTION




# =========================================================
# GLOBAL PARAMETERS
# =========================================================
xmin, xmax, ymin, ymax = 541550, 584200, 8602400, 8667100
resolution = 25
plot_results = False

target_crs = "EPSG:32633"
plots = True
borne_plot = 200 # vmin and vmax for diff

study_area = Polygon([
    (xmin, ymin),
    (xmin, ymax),
    (xmax, ymax),
    (xmax, ymin)
])


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def build_regular_grid(xmin, xmax, ymin, ymax, res):
    """Create a regular UTM grid and corresponding affine transform."""
    nx = int((xmax - xmin) / res)
    ny = int((ymax - ymin) / res)

    transform = from_bounds(xmin, ymin, xmax, ymax, nx, ny)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymax, ymin, ny)

    return transform, (ny, nx), x, y


def reproject_raster_to_grid(src_array, src_transform, src_crs,
                             dst_transform, dst_shape,
                             dst_crs="EPSG:32633",
                             method=Resampling.bilinear):
    """Reproject any raster to a common grid."""
    output = np.full(dst_shape, np.nan, dtype=float)

    reproject(
        source=src_array,
        destination=output,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=method
    )

    return output


def rasterize_vector_mask(gdf, transform, shape):
    """Rasterize vector polygons into a binary mask."""
    shapes = [(geom, 1) for geom in gdf.geometry]

    return rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False
    )


def load_and_crop_tif(path, xmin, xmax, ymin, ymax):
    """
    Load and crop a GeoTIFF to a bounding box,
    while updating the affine transform correctly.
    """

    with rasterio.open(path) as src:

        array = src.read(1).astype(float)

        nodata = src.nodata
        crs = src.crs
        res_x, res_y = src.res

        if nodata is not None:
            array[array == nodata] = np.nan

        # Original coordinates
        x = np.arange(src.width) * res_x + src.bounds.left
        y = src.bounds.top - np.arange(src.height) * abs(res_y)

        # Subset indices
        ix = np.where((x >= xmin) & (x <= xmax))[0]
        iy = np.where((y >= ymin) & (y <= ymax))[0]

        # Crop array
        cropped = array[np.ix_(iy, ix)]

        # Subset coordinates
        x_sub = x[ix]
        y_sub = y[iy]

        # NEW transform for cropped raster
        transform_sub = rasterio.transform.from_origin(
            x_sub.min(),
            y_sub.max(),
            res_x,
            abs(res_y)
        )

    return cropped, transform_sub, crs


def plot_beds(bed, name):
    plt.figure(figsize=(12,10))
    plt.imshow(bed, cmap="terrain", origin="upper",vmin = 20, vmax=600)
    plt.colorbar(label="Altitude (m)")
    plt.title(f"Bedrock {name}")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.show()

def plot_beds_side_by_side(beds, names):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    for ax, bed, name in zip(axes, beds, names):
        im = ax.imshow(
            bed,
            cmap="terrain",
            origin="upper",
            vmin=20,
            vmax=600
        )

        ax.set_title(f"Bedrock {name}")
        ax.set_xlabel("Easting (px)")
        ax.set_ylabel("Northing (px)")

    # colorbar globale (important pour comparaison)
    cbar = fig.colorbar(im, ax=axes, shrink=0.7, label="Altitude (m)")

    #plt.tight_layout()
    plt.savefig("All_bedrocks.jpg", dpi = 200) 
    plt.show()

def plot_diff_side_by_side(diffs, names, borne):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    for ax, diff, name in zip(axes, diffs, names):
        im = ax.imshow(
            diff,
            cmap="bwr",
            origin="upper",
            vmin= - borne,
            vmax= borne
        )

        ax.set_title(f"Difference {name}")
        ax.set_xlabel("Easting (px)")
        ax.set_ylabel("Northing (px)")

    # colorbar globale (important pour comparaison)
    cbar = fig.colorbar(im, ax=axes, shrink=0.7, label="Altitude (m)")

    #plt.tight_layout()
    plt.savefig("Differences_between_bedrocks.jpg", dpi = 200)
    plt.show()


# =========================================================
# 1. STUDY AREA & GLACIER MASK
# =========================================================
region_path = "~/PhD_Lucie/DATA/Maps/Shapefiles_regions/mask_heer_land.shp"
gdf_region = gpd.read_file(region_path).to_crs(target_crs)
region_geom = gdf_region.unary_union

glacier_path = "~/PhD_Lucie/DATA/GLACIER_OUTLINES/CryoClim_GAO_SJ_1936-1972/CryoClim_GAO_SJ_1936-1972.shp"
gdf_glaciers = gpd.read_file(glacier_path).to_crs(target_crs)

gdf_glaciers = gdf_glaciers[gdf_glaciers.intersects(region_geom)]



# =========================================================
# 2. COMMON GRID
# =========================================================
grid_transform, grid_shape, x_grid, y_grid = build_regular_grid(
    xmin, xmax, ymin, ymax, resolution
)


# =========================================================
# 3. FURST BEDROCK
# =========================================================
furst_path = "~/PhD_Lucie/DATA/BEDROCK/Furst-svift-topo/"


ds_furst = xr.open_dataset(
    os.path.join(furst_path, "svift_v1_bed.nc")
)

bed_furst = ds_furst["bed"].sel(
    x=slice(xmin, xmax),
    y=slice(ymin, ymax)
)

# Convert to numpy
bed_furst_array = bed_furst.values.astype(float)

# Coordinates
x_furst = bed_furst["x"].values
y_furst = bed_furst["y"].values

# Grid spacing
dx_furst = x_furst[1] - x_furst[0]
dy_furst = y_furst[1] - y_furst[0]

print("Furst dx:", dx_furst)
print("Furst dy:", dy_furst)

# Build affine transform directly from coordinates
furst_transform = rasterio.transform.from_origin(
    x_furst.min(),
    y_furst.max(),
    abs(dx_furst),
    abs(dy_furst)
)

# Reproject to common grid
bed_furst_grid = reproject_raster_to_grid(
    bed_furst_array,
    furst_transform,
    "EPSG:32633",
    grid_transform,
    grid_shape
)

print("Furst reprojection:")
print("min =", np.nanmin(bed_furst_grid))
print("max =", np.nanmax(bed_furst_grid))

bed_furst_grid = np.flipud(bed_furst_grid)
plot_beds(bed_furst_grid, "Furst")

# =========================================================
# 4. VAN PELT BEDROCK
# =========================================================
vp_path = "../../PhD_Lucie/DATA/BEDROCK/Van_pelt/Bed_map.tif"

with rasterio.open(vp_path) as src:
    bed_vp = src.read(1).astype(float)
    vp_transform = src.transform
    vp_crs = src.crs
    nodata = src.nodata
    vp_bounds = src.bounds
    vp_nx, vp_ny = src.width, src.height
    vp_res = src.res[0]

if nodata is not None:
    bed_vp[bed_vp == nodata] = np.nan

bed_vp_sub = load_and_crop_tif(vp_path, xmin, xmax, ymin, ymax)

bed_vp_grid = reproject_raster_to_grid(
    bed_vp,
    vp_transform,
    vp_crs,
    grid_transform,
    grid_shape
)


# =========================================================
# 5. FRANCK BEDROCK
# =========================================================
franck_path = "mosaic_heer_land_franck.tif"

bed_franck, franck_transform, franck_crs = load_and_crop_tif(
    franck_path, xmin, xmax, ymin, ymax
)

bed_franck_grid = reproject_raster_to_grid(
    bed_franck,
    franck_transform,
    franck_crs,
    grid_transform,
    grid_shape
)



# =========================================================
# 6. GLACIER MASK (COMMON GRID)
# =========================================================
glacier_mask = rasterize_vector_mask(
    gdf_glaciers,
    grid_transform,
    grid_shape
)


# =========================================================
# 7. APPLY MASK
# =========================================================
bed_furst_masked = np.where(glacier_mask == 0, np.nan, bed_furst_grid)
bed_vp_masked = np.where(glacier_mask == 0, np.nan, bed_vp_grid)
bed_franck_masked = np.where(glacier_mask == 0, np.nan, bed_franck_grid)

if plot_results :

    plot_beds(bed_furst_masked, "Fürst")
    plot_beds(bed_vp_masked, "Van Pelt")
    plot_beds(bed_franck_masked, "Franck")

plot_beds_side_by_side(
    [bed_furst_masked, bed_vp_masked, bed_franck_masked],
    ["Fürst", "Van Pelt", "Franck"]
)




# =========================================================
# 8. DIFFERENCE MAPS
# =========================================================
diff_furst_vp = bed_furst_masked - bed_vp_masked
diff_franck_vp = bed_franck_masked - bed_vp_masked
diff_furst_franck = bed_furst_masked - bed_franck_masked


plot_diff_side_by_side(
    [diff_furst_vp, diff_franck_vp, diff_furst_franck],
    ["Fürst - Van Pelt ", "Franck - Van Pelt", "Fürst -Franck"], 200
)


# =========================================================
# 9. VISUALIZATION
# =========================================================
if plot_results:

    plt.figure(figsize=(12,10))
    plt.imshow(diff_furst_vp, cmap="bwr", origin="upper", vmin = - borne_plot, vmax= borne_plot)
    plt.colorbar(label="Elevation difference (m)")
    plt.title("Furst - Van Pelt bedrock difference")
    plt.show()

    plt.figure(figsize=(12,10))
    plt.imshow(diff_franck_vp, cmap="bwr", origin="upper", vmin = - borne_plot, vmax= borne_plot)
    plt.colorbar(label="Elevation difference (m)")
    plt.title("Franck - Van Pelt bedrock difference")
    plt.show()

    plt.figure(figsize=(12,10))
    plt.imshow(diff_furst_franck, cmap="bwr", origin="upper", vmin = - borne_plot, vmax= borne_plot)
    plt.colorbar(label="Elevation difference (m)")
    plt.title("Furst - Franck bedrock difference")
    plt.show()


# =========================================================
# 9. Save as NCDF
# =========================================================

def save_bedrock_to_netcdf(bedrock_array,
                           x_coords,
                           y_coords,
                           resolution,
                           name,
                           output_dir="Datasets"):
    """
    Save a bedrock array to NetCDF.
    """

    ds = xr.Dataset(
        {
            "topg": (("y", "x"), bedrock_array)
        },
        coords={
            "x": x_coords,
            "y": y_coords
        }
    )

    ds["topg"].attrs = {
        "long_name": "Bedrock elevation",
        "units": "m"
    }

    ds.attrs = {
        "projection": "EPSG:32633",
        "resolution_m": resolution,
        "dataset": name
    }

    output_path = os.path.join(
        output_dir,
        f"Bedrock_{name}_{resolution}m.nc"
    )

    ds.to_netcdf(output_path)

    print(f"Saved: {output_path}")


save_bedrock_to_netcdf(
    bed_furst_grid,
    x_grid,
    y_grid,
    resolution,
    "Furst"
)

save_bedrock_to_netcdf(
    bed_vp_grid,
    x_grid,
    y_grid,
    resolution,
    "VanPelt"
)

save_bedrock_to_netcdf(
    bed_franck_grid,
    x_grid,
    y_grid,
    resolution,
    "Franck"
)
