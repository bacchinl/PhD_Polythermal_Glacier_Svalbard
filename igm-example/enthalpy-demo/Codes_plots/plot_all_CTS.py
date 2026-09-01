import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.interpolate import RegularGridInterpolator, splprep, splev
import imageio


# =====================================================
# PARAMETERS
# =====================================================

smooth = False
MAKE_GIF = False
GIF_FPS = 4

point_fin = 10e9

plot_obs = True
plot_ela = False
mask_for_obs = False

outputs_path = "../multirun"

if len(sys.argv) > 1:
    # Argument
    date_simu = sys.argv[1]
else:
    # Most recent simulation in outputs
    simulations = [
        path for path in glob.glob(os.path.join(outputs_path, "*", "*"))
        if os.path.isdir(path)
    ]

    if not simulations:
        raise FileNotFoundError(f"No simulation output.nc found in {outputs_path}")

    latest_simu = max(simulations, key=os.path.getmtime)
    date_simu = os.path.relpath(latest_simu, outputs_path)


# =====================================================
# PATHS
# =====================================================
simu_path = os.path.join(outputs_path, date_simu)

data_dir = "../obs_cts/thickness_cts_csv"
glacier_name = "dronbreen"
year_obs = 2022

# Tous les CSV correspondant au glacier et à l'année
csv_pattern = os.path.join(
    data_dir,
    f"thickness_cts_points_{glacier_name}-{year_obs}*.csv"
)

csv_files = sorted(glob.glob(csv_pattern))

if len(csv_files) == 0:
    raise FileNotFoundError(
        f"No file found with pattern : {csv_pattern}"
    )


# =====================================================
# OUTPUT DIRECTORY
# =====================================================
# Plots are stored in :
# .../Plots/Ice_type/
out_dir = os.path.join(simu_path, "Plots", "Ice_type")
os.makedirs(out_dir, exist_ok=True)


# =====================================================
# LOAD MODEL OUTPUT
# =====================================================
print("Loading model output...")

ds = xr.open_dataset(os.path.join(simu_path, "output.nc"))

E = ds["E"]
Epmp = ds["E_pmp"]
thk = ds["thk"]
usurf = ds["usurf"]
topg = ds["topg"]

if plot_ela:
    ela = ds["ela"]

x = ds["x"].values
y = ds["y"].values
z = ds["z"].values
time = ds["time"].values

ntime, nz = len(time), len(z)

vert_spacing = 4
Nz = nz

zeta_edges = np.arange(Nz + 1) / Nz
zeta_edges = (
    zeta_edges / vert_spacing
) * (
    1.0 + (vert_spacing - 1.0) * zeta_edges
)

zeta_mid = 0.5 * (
    zeta_edges[:-1] + zeta_edges[1:]
)


# =====================================================
# COLORMAP ICE TYPE
# =====================================================
cmap_ice = ListedColormap(["lightblue", "salmon"])
norm_ice = BoundaryNorm(
    [-0.5, 0.5, 1.5],
    cmap_ice.N
)


# =====================================================
# LOOP OVER CSV FILES
# =====================================================
for csv_file in csv_files:

    print("\n" + "=" * 70)
    print("Processing:", csv_file)

    # -------------------------------------------------
    # RADAR LINE NAME
    # -------------------------------------------------
    radar_line = (
        os.path.basename(csv_file)
        .replace("thickness_cts_points_", "")
        .replace(".csv", "")
    )

    print("Radar line:", radar_line)

    # -------------------------------------------------
    # OUTPUT DIRECTORY FOR THIS RADAR LINE
    # -------------------------------------------------
    radar_out_dir = os.path.join(
        out_dir,
        radar_line
    )

    
    # -------------------------------------------------
    # LOAD GPR DATA
    # -------------------------------------------------
    if plot_obs:

        columns_to_keep = [
            "radar_key",
            "distance",
            "easting",
            "northing",
            "temperate_elevation",
            "bed_elevation"
        ]

        gpr_df = pd.read_csv(
            csv_file,
            usecols=columns_to_keep
        )

        
        gpr_df_trace = gpr_df[
            gpr_df["radar_key"] == radar_line
        ].copy()

    
        if len(gpr_df_trace) == 0:
            print(
                f"WARNING: aucun radar_key = {radar_line}"
            )
            print("Le fichier est ignoré.")
            continue

        gpr_df_trace = gpr_df_trace.dropna(
            subset=[
                "easting",
                "northing",
                "temperate_elevation",
                "bed_elevation"
            ]
        )

        x_flow = gpr_df_trace["easting"].values
        y_flow = gpr_df_trace["northing"].values
        cts_obs = gpr_df_trace[
            "temperate_elevation"
        ].values
        bed_obs = gpr_df_trace[
            "bed_elevation"].values


        # -------------------------------------------------
        # SMOOTH FLOWLINE
        # -------------------------------------------------
        if smooth:

            if len(x_flow) < 4:
                print(
                    f"WARNING: not enough points to smooth "
                    f" {radar_line}"
                )
                continue

            tck, _ = splprep(
                np.vstack([x_flow, y_flow]),
                s=0,
                k=min(3, len(x_flow) - 1)
            )

            u = np.linspace(0, 1, 200)

            x_flow, y_flow = splev(
                u,
                tck
            )

            
            dist_flow_original = np.concatenate([
                [0],
                np.cumsum(
                    np.sqrt(
                        np.diff(
                            gpr_df_trace["easting"].values
                        ) ** 2
                        +
                        np.diff(
                            gpr_df_trace["northing"].values
                        ) ** 2
                    )
                )
            ])

            dist_flow_smooth = np.concatenate([
                [0],
                np.cumsum(
                    np.sqrt(
                        np.diff(x_flow) ** 2
                        +
                        np.diff(y_flow) ** 2
                    )
                )
            ])

            # obs interp on flowline
            cts_obs = np.interp(
                dist_flow_smooth,
                dist_flow_original,
                gpr_df_trace[
                    "temperate_elevation"
                ].values
            )

            dist_flow = dist_flow_smooth

        else:

            dist_flow = np.concatenate([
                [0],
                np.cumsum(
                    np.sqrt(
                        np.diff(x_flow) ** 2
                        +
                        np.diff(y_flow) ** 2
                    )
                )
            ])

        # -------------------------------------------------
        # DISTANCE ALONG FLOWLINE
        # -------------------------------------------------
        mask_flow = dist_flow <= point_fin

        dist_km = dist_flow[mask_flow] / 1000.0

        cts_obs = cts_obs[mask_flow]
        bed_obs = bed_obs[mask_flow]


        dist_max = np.max(dist_km)

        
        # -------------------------------------------------
        # OPTIONAL OBSERVATION MASK
        # -------------------------------------------------
        if mask_for_obs:

            mask_obs = (
                (dist_km <= point_fin_obs)
                &
                (dist_km >= point_deb_obs)
            )

            dist_obs = dist_km[mask_obs]
            cts_obs_plot = cts_obs[mask_obs]

        else:

            dist_obs = dist_km
            cts_obs_plot = cts_obs

    else:

        # Si plot_obs = False, still need flowline
        raise ValueError(
            "plot_obs=False n'est pas compatible avec "
            "le traitement automatique des CSV."
        )


    # =================================================
    # INTERPOLATION ALONG FLOWLINE
    # =================================================
    E3d = np.full(
        (ntime, nz, len(x_flow)),
        np.nan
    )

    Epmp3d = np.full_like(E3d, np.nan)

    thk_f = np.full(
        (ntime, len(x_flow)),
        np.nan
    )

    usurf_f = np.full_like(thk_f, np.nan)
    topg_f = np.full_like(thk_f, np.nan)

    

    pts2d = np.vstack([
        y_flow,
        x_flow
    ]).T

    for it in range(ntime):

        interp_E = RegularGridInterpolator(
            (z, y, x),
            E.isel(time=it).values
        )

        interp_Epmp = RegularGridInterpolator(
            (z, y, x),
            Epmp.isel(time=it).values
        )

        interp_thk = RegularGridInterpolator(
            (y, x),
            thk.isel(time=it).values
        )

        interp_usurf = RegularGridInterpolator(
            (y, x),
            usurf.isel(time=it).values
        )

        interp_topg = RegularGridInterpolator(
            (y, x),
            topg.isel(time=it).values
        )

        thk_f[it] = interp_thk(pts2d)
        usurf_f[it] = interp_usurf(pts2d)
        topg_f[it] = interp_topg(pts2d)

        for k in range(nz):

            pts3d = np.vstack([
                np.full_like(x_flow, z[k]),
                y_flow,
                x_flow
            ]).T

            E3d[it, k] = interp_E(pts3d)
            Epmp3d[it, k] = interp_Epmp(pts3d)

    

    # =================================================
    # ICE TYPE
    # =================================================
    # 0 = cold
    # 1 = temperate
    ice_type_3d = np.zeros_like(E3d)

    ice_type_3d[E3d >= Epmp3d] = 1


    # =================================================
    # COLD LAYER THICKNESS
    # =================================================
    z_if = np.zeros(nz + 1)

    z_if[1:-1] = 0.5 * (
        z[:-1] + z[1:]
    )

    z_if[0] = (
        z[0]
        - (z[1] - z[0]) / 2
    )

    z_if[-1] = (
        z[-1]
        + (z[-1] - z[-2]) / 2
    )

    dz = np.diff(z_if)
    fractions = dz / dz.sum()

    cold_thickness = np.full(
        (ntime, len(x_flow)),
        np.nan
    )

    for it in range(ntime):

        layer_thk = (
            fractions[:, None]
            * thk_f[it]
        )

        cold_mask = (
            ice_type_3d[it] == 0
        )

        cold_thickness[it] = np.nansum(
            layer_thk * cold_mask,
            axis=0
        )


    # =================================================
    # STATIC PLOT / GIF
    # =================================================
    frames = []

    indices = (
        range(ntime)
        if MAKE_GIF
        else [-1]
    )

    for it in indices:

        ice_crop = ice_type_3d[it,:,mask_flow]
        
        if ice_crop.shape != x.shape:
            ice_crop = ice_crop.T

        us = usurf_f[it,mask_flow]

        bg = topg_f[it,mask_flow]

        thk_current = thk_f[it,mask_flow]

        if plot_ela:

            ela_it = float(
                ela[it]
            )

        X = np.tile(
            dist_km,
            (nz, 1)
        )

        Z = (
            bg[None, :]
            +
            zeta_mid[:, None]
            *
            (
                us - bg
            )[None, :]
        )

        # -------------------------------------------------
        # PLOT
        # -------------------------------------------------
        plt.figure(
            figsize=(11, 5)
        )

        plt.contourf(
            X,
            Z,
            ice_crop,
            levels=[
                -0.5,
                0.5,
                1.5
            ],
            cmap=cmap_ice,
            norm=norm_ice
        )

        # -------------------------------------------------
        # OBSERVED CTS
        # -------------------------------------------------
        if plot_obs:

            cts_obs_plot_current = cts_obs_plot.copy()

            bg_obs = np.interp(dist_obs,dist_km,bg)

            us_obs = np.interp(dist_obs,dist_km,us)

            valid_obs = ((cts_obs_plot_current >= bg_obs)&(cts_obs_plot_current <= us_obs)&(cts_obs_plot_current > bed_obs))

            cts_obs_plot_current = np.where(
                valid_obs,
                cts_obs_plot_current,
                np.nan
            )

            plt.plot(
                dist_obs,
                cts_obs_plot_current,
                "k--",
                lw=2,
                label="Observed CTS"
            )

        # -------------------------------------------------
        # SURFACE
        # -------------------------------------------------
        plt.plot(
            dist_km,
            us,
            "k",
            lw=1.5
        )

        # -------------------------------------------------
        # BED
        # -------------------------------------------------
        plt.plot(
            dist_km,
            bg,
            "k--",
            lw=1.2
        )

        # -------------------------------------------------
        # ELA
        # -------------------------------------------------
        if plot_ela:

            x_ela = np.argmin(
                np.abs(us - ela_it)
            )

            plt.plot(
                [
                    dist_km[x_ela],
                    dist_km[x_ela]
                ],
                [
                    ela_it - 10,
                    ela_it + 10
                ],
                "k-",
                lw=2
            )

        # -------------------------------------------------
        # COLORBAR
        # -------------------------------------------------
        cbar = plt.colorbar(
            ticks=[0, 1]
        )

        cbar.ax.set_yticklabels([
            "Cold ice",
            "Temperate ice"
        ])

        # -------------------------------------------------
        # BEDROCK
        # -------------------------------------------------
        z_min = np.min(Z)

        plt.fill_between(
            dist_km,
            bg,
            z_min,
            color="saddlebrown",
            alpha=0.8,
            zorder=2,
            label="Bedrock"
        )

        # -------------------------------------------------
        # LABELS
        # -------------------------------------------------
        plt.xlabel(
            "Distance along flowline (km)",
            fontsize=14
        )

        plt.ylabel(
            "Altitude (m a.s.l.)",
            fontsize=14
        )

        plt.xticks(
            fontsize=14
        )

        plt.yticks(
            fontsize=14
        )

        plt.title(
            f"Ice type vertical section – "
            f"{radar_line}",
            fontsize=16
        )

        plt.grid(
            True,
            linestyle=":"
        )

        plt.tight_layout()

        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------
        fname = os.path.join(
            out_dir,
            f"{radar_line}_ice_type_{time[it]}.png"
        )

        plt.savefig(
            fname,
            dpi=200
        )

        plt.close()

        print(
            "Saved:",
            fname
        )

        if MAKE_GIF:
            frames.append(fname)


    # =================================================
    # GIF
    # =================================================
    if MAKE_GIF:

        gif_path = os.path.join(
            radar_out_dir,
            f"{radar_line}_ice_type.gif"
        )

        with imageio.get_writer(
            gif_path,
            mode="I",
            fps=GIF_FPS
        ) as writer:

            for f in frames:
                writer.append_data(
                    imageio.imread(f)
                )

        print(
            "GIF created:",
            gif_path
        )


print("\nAll radar lines processed.")
