import os
import glob

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator

# =========================================================
# PARAMETERS
# =========================================================

resolution = 25

data_dir = os.path.expanduser(
    "~/PhD_Lucie/DATA/GPR/Mannerfelt/thickness_cts_points_csv"
)

glacier_name = "edvardbreen"

path_bedrocks = "Datasets"

output_dir = f"Plots/{glacier_name}"
os.makedirs(output_dir, exist_ok=True)


# =========================================================
# NETCDF FILE PATHS
# =========================================================

furst_nc_path = os.path.join(
    path_bedrocks,
    f"Bedrock_Furst_{resolution}m.nc"
)

vp_nc_path = os.path.join(
    path_bedrocks,
    f"Bedrock_VanPelt_{resolution}m.nc"
)

franck_nc_path = os.path.join(
    path_bedrocks,
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
# INTERPOLATORS
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


# =========================================================
# INITIATE TEXT STATS
# =========================================================
stats_file = os.path.join(
    output_dir,
    f"stats_{glacier_name}.txt"
)

fstats = open(stats_file, "w")

fstats.write(f"Statistics for glacier {glacier_name}\n")
fstats.write("=" * 60 + "\n\n" ) ## fancy headline

# =========================================================
# FIND ALL CSV FILES FOR THE GLACIER
# =========================================================

csv_pattern = os.path.join(
    data_dir,
    f"thickness_cts_points_{glacier_name}*.csv"
)

csv_files = sorted(glob.glob(csv_pattern))

print(f"\nFound {len(csv_files)} radar lines for {glacier_name}")

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
    fstats.write("=" * 60 + "\n\n")
    fstats.write(f"   Radar line: {radar_line}\n\n")
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

    # sécurité si plusieurs radar_key existent
    gpr_df_trace = gpr_df[
        gpr_df["radar_key"] == radar_line
    ].copy()

    if len(gpr_df_trace) == 0:
        print("No matching radar_key found.")
        continue

    # =====================================================
    # GPR COORDINATES
    # =====================================================

    x_gpr = gpr_df_trace["easting"].values
    y_gpr = gpr_df_trace["northing"].values

    points_gpr = np.column_stack(
        (y_gpr, x_gpr)
    )

    # =====================================================
    # INTERPOLATE BEDROCKS
    # =====================================================

    bed_furst_along_line = interp_furst(points_gpr)
    bed_vp_along_line = interp_vp(points_gpr)
    bed_franck_along_line = interp_franck(points_gpr)

    bed_gpr_obs = gpr_df_trace["bed_elevation"].values
    surf_gpr_obs =gpr_df_trace["elevation"].values
    distance = gpr_df_trace["distance"].values

    # =====================================================
    # STATISTICS
    # =====================================================

    diff_furst = bed_furst_along_line - bed_gpr_obs
    diff_vp = bed_vp_along_line - bed_gpr_obs
    diff_franck = bed_franck_along_line - bed_gpr_obs

    mean_error_furst = np.nanmean(diff_furst)
    mean_error_vp = np.nanmean(diff_vp)
    mean_error_franck = np.nanmean(diff_franck)

    std_furst = np.nanstd(diff_furst)
    std_vp = np.nanstd(diff_vp)
    std_franck = np.nanstd(diff_franck)

    stats_text = (
        f"MEAN ERROR FURST    : {mean_error_furst:.2f} m\n"
        f"MEAN ERROR VAN PELT : {mean_error_vp:.2f} m\n"
        f"MEAN ERROR FRANCK   : {mean_error_franck:.2f} m\n\n"
        f"STANDRAD DEV FURST      : {std_furst:.2f} m\n"
        f"STANDARD DEV VAN PELT   : {std_vp:.2f} m\n"
        f"STANDARD DEV FRANCK     : {std_franck:.2f} m\n\n"
    )

    print(stats_text)
    fstats.write(stats_text)
    # =====================================================
    # PLOT
    # =====================================================

    plt.figure(figsize=(12, 6))

    plt.plot(
        distance,
        bed_gpr_obs,
        linewidth=3,
        label="GPR observed bed"
    )

    plt.plot(
        distance,
        surf_gpr_obs,
        linewidth=3,
        color = "k",
        label="GPR observed surface"
    )


    plt.plot(
        distance,
        bed_furst_along_line,
        label=(
            f"Furst "
            f"(bias={mean_error_furst:.2f} m, "
            f"std={std_furst:.2f} m)"
        )
    )

    plt.plot(
        distance,
        bed_vp_along_line,
        label=(
            f"Van Pelt "
            f"(bias={mean_error_vp:.2f} m, "
            f"std={std_vp:.2f} m)"
        )
    )

    plt.plot(
        distance,
        bed_franck_along_line,
        label=(
            f"Franck "
            f"(bias={mean_error_franck:.2f} m, "
            f"std={std_franck:.2f} m)"
        )
    )

    plt.xlabel("Distance along track (m)")
    plt.ylabel("Elevation (m a.s.l.)")
    plt.title(
        f"Bedrock comparison along radar line\n{radar_line}"
    )

    plt.grid(True)
    plt.legend()

    output_file = os.path.join(
        output_dir,
        f"Bedrock_comparison_{radar_line}.jpg"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_file}")

fstats.close()
print(f"\nStatistics saved in {stats_file}")
print("\nDone.")
