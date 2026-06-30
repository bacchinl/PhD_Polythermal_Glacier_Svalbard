import os
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# =========================================================
# PARAMETERS
# =========================================================
resolution = 50

data_dir = "~/PhD_Lucie/DATA/GPR/Mannerfelt/"

radar_line = "mettebreen-20230305-DAT_0235_A1_8"

# =========================================================
# NETCDF FILE PATHS
# =========================================================
furst_nc_path = os.path.join(
    ".",
    f"Bedrock_Furst_{resolution}m.nc"
)

vp_nc_path = os.path.join(
    ".",
    f"Bedrock_VanPelt_{resolution}m.nc"
)

franck_nc_path = os.path.join(
    ".",
    f"Bedrock_Franck_{resolution}m.nc"
)


# =========================================================
# LOAD BEDROCK DATASETS
# =========================================================
ds_furst = xr.open_dataset(furst_nc_path)
ds_vp = xr.open_dataset(vp_nc_path)
ds_franck = xr.open_dataset(franck_nc_path)

print("Furst dataset:")
print(ds_furst)

print("\nVan Pelt dataset:")
print(ds_vp)

print("\nFranck dataset:")
print(ds_franck)


# =========================================================
# EXTRACT BEDROCK ARRAYS
# =========================================================
bed_furst = ds_furst["topg"].values
bed_vp = ds_vp["topg"].values
bed_franck = ds_franck["topg"].values

x = ds_furst["x"].values
y = ds_furst["y"].values


# =========================================================
# LOAD GPR CSV DATA
# =========================================================
gpr_csv_path = os.path.join(
    data_dir,
    "mannerfelt_ragnamarie_mette.csv"
)

# Keep only selected columns
columns_to_keep = [
    "radar_key", "distance",
    "easting", "northing", "bed_elevation"
]

gpr_df = pd.read_csv(
    gpr_csv_path,
    usecols=columns_to_keep
)

gpr_df_trace = gpr_df[gpr_df["radar_key"] == radar_line].copy()




print("\nGPR dataframe:")
print(gpr_df_trace.head())


x_gpr = gpr_df_trace["easting"].values
y_gpr = gpr_df_trace["northing"].values

# =========================================================
# SELECT TRACE IN DS
# =========================================================

interp_furst = RegularGridInterpolator(
    (y, x),
    bed_furst,
    bounds_error=False,
    fill_value=np.nan
)

interp_vp = RegularGridInterpolator(
    (y, x),
    bed_vp,
    bounds_error=False,
    fill_value=np.nan
)

interp_franck = RegularGridInterpolator(
    (y, x),
    bed_franck,
    bounds_error=False,
    fill_value=np.nan
)

points_gpr = np.column_stack((y_gpr, x_gpr))

bed_furst_along_line = interp_furst(points_gpr)

bed_vp_along_line = interp_vp(points_gpr)

bed_franck_along_line = interp_franck(points_gpr)


bed_gpr_obs = gpr_df_trace["bed_elevation"].values
distance = gpr_df_trace["distance"].values

mean_error_furst = np.mean(bed_furst_along_line-bed_gpr_obs)
mean_error_vp = np.mean(bed_vp_along_line-bed_gpr_obs)
mean_error_franck = np.mean(bed_franck_along_line-bed_gpr_obs)

print("ERROR FURST ", mean_error_furst, "m")
print("ERROR VanPelt ", mean_error_vp, "m")
print("ERROR FRANCK ", mean_error_franck, "m")



# =========================================================
# PLOT
# =========================================================


plt.figure(figsize=(12,6))

plt.plot(
    distance,
    bed_gpr_obs,
    label="GPR observations",
    linewidth=3
)

plt.plot(
    distance,
    bed_furst_along_line,
    label=f"Furst, mean error : {mean_error_furst}m"
)

plt.plot(
    distance,
    bed_vp_along_line,
    label=f"Van Pelt, mean error : {mean_error_vp}m"
)

plt.plot(
    distance,
    bed_franck_along_line,
    label=f"Franck, mean error : {mean_error_franck}m"
)

plt.xlabel("Distance along profile (m)")
plt.ylabel("Bed elevation (m)")
plt.title(f"Bedrock comparison along radar line {radar_line}")

plt.legend()
plt.grid()

plt.savefig(f"Plots/Bedrock_comparaison_{radar_line}.jpg")
plt.show()
