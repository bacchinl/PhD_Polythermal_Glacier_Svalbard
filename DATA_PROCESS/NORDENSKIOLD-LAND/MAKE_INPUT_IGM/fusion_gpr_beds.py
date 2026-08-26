import xarray as xr
import rioxarray
import numpy as np
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.io import MemoryFile
import matplotlib.pyplot as plt
# ============================================================================
# Paramètres
# ============================================================================

glaciers = ["mettebreen", "kroppbreen", "vallakrabreen", "edvardbreen", "ragna_mariebreen"]

input_dir = "BEDS_FROM_GPR"

# CRS (à adapter)
crs = "EPSG:32633"

# Domaine souhaité
xmin = 543500
xmax = 566500
ymin = 8633000
ymax = 8655000

resolution = 50  # mètres

output_file = "Bedrock_global_50m.nc"

# ============================================================================
# Construction de la grille cible
# ============================================================================

x50 = np.arange(xmin, xmax + 50, 50)
y50 = np.arange(ymax, ymin - 50, -50)

mosaic = xr.DataArray(
    np.full((len(y50), len(x50)), np.nan),
    coords={"y": y50, "x": x50},
    dims=("y", "x"),
)

for glacier in glaciers:

    ds = xr.open_dataset(f"{input_dir}/Input_{glacier}_only_GPR_res_20m.nc")

    bed = ds["topg"]

    if bed.x[0] > bed.x[-1]:
        bed = bed.sortby("x")

    if bed.y[0] < bed.y[-1]:
        bed = bed.sortby("y", ascending=False)


    bed50 = bed.interp(
        x=x50,
        y=y50,
        method="nearest",
    )

    mosaic = mosaic.combine_first(bed50)

mosaic.to_dataset(name="topg").to_netcdf("Bedrock_global_50m.nc")


ds_out = xr.Dataset(
    {
        "topg": (("y","x"), mosaic),
    },
    coords={
        "x": (("x",), x50),
        "y": (("y",), y50)
    }
)

print("Fini :", output_file)
