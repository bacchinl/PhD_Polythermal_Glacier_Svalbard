import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.interpolate import RegularGridInterpolator


# ============================================================
# INITIALIZE
# ============================================================

def initialize(cfg, state):

    data_dir = os.path.expanduser(
        cfg.outputs.plot_CTS.data_dir
    )

    year_obs = cfg.outputs.plot_CTS.year_obs
    glacier_name = cfg.outputs.plot_CTS.glacier_name

    csv_pattern = os.path.join(
        data_dir,
        f"thickness_cts_points_{glacier_name}-{year_obs}*.csv"
    )

    csv_files = sorted(
        glob.glob(csv_pattern)
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CTS files found with pattern:\n{csv_pattern}"
        )



    state.gpr_profiles_to_plot = []

    for csv_file in csv_files:

        radar_line = (
            os.path.basename(csv_file)
            .replace("thickness_cts_points_", "")
            .replace(".csv", "")
        )

        columns_to_keep = [
            "radar_key",
            "easting",
            "northing",
            "temperate_elevation",
            "bed_elevation",
        ]

        gpr_df = pd.read_csv(
            csv_file,
            usecols=columns_to_keep,
        )

        state.gpr_profiles_to_plot.append(
            {
                "name": radar_line,
                "x": gpr_df["easting"].values,
                "y": gpr_df["northing"].values,
                "cts_obs": gpr_df["temperate_elevation"].values,
                "bed_obs": gpr_df["bed_elevation"].values,
            }
        )

    

# ============================================================
# RUN
# ============================================================

def run(cfg, state):

    # Same logic as plot2d:
    # only generate output when IGM asks for a saved result.
    if not state.saveresult:
        return

    
    for profile in state.gpr_profiles_to_plot:

        plot_profile(
            cfg,
            state,
            profile,
        )


# ============================================================
# FINALIZE
# ============================================================

def finalize(cfg, state):
    pass


# ============================================================
# PLOT PROFILE
# ============================================================

def plot_profile(cfg, state, profile):
    """
    Plot modelled ice type along an observed CTS radar line
    for the CURRENT IGM model state.
    """

    # ========================================================
    # OPTIONS
    # ========================================================

    plot_obs = cfg.outputs.plot_CTS.plot_obs
    plot_ela = cfg.outputs.plot_CTS.plot_ela

    # ========================================================
    # RADAR LINE NAME
    # ========================================================

    radar_line = profile["name"]

    
    # ========================================================
    # CURRENT MODEL STATE
    # ========================================================

    # IMPORTANT:
    # state contains the CURRENT timestep.
    # There is therefore NO [it] here.

    E = np.asarray(state.E)
    Epmp = np.asarray(state.E_pmp)

    thk = np.asarray(state.thk)
    usurf = np.asarray(state.usurf)
    topg = np.asarray(state.topg)

    x = np.asarray(state.x)
    y = np.asarray(state.y)
    
    nz = E.shape[0]
    z = np.arange(nz)

    # ========================================================
    # CURRENT TIME
    # ========================================================

    time = getattr(state, "t", 0)

    if hasattr(time, "numpy"):
        time = time.numpy()

    time = np.asarray(time).squeeze()

    # Convert scalar numpy array to Python scalar
    if np.ndim(time) == 0:
        time = time.item()

    # ========================================================
    # OBSERVATIONS
    # ========================================================

    cts_obs = np.asarray(profile["cts_obs"])
    bed_obs = np.asarray(profile["bed_obs"])

    x_flow = np.asarray(profile["x"])
    y_flow = np.asarray(profile["y"])

    # ========================================================
    # DISTANCE ALONG RADAR LINE
    # ========================================================

    dist_flow = np.concatenate(
        [
            [0.0],
            np.cumsum(
                np.sqrt(
                    np.diff(x_flow) ** 2
                    + np.diff(y_flow) ** 2
                )
            ),
        ]
    )

    dist_km = dist_flow / 1000.0

    # ========================================================
    # INTERPOLATE 2D MODEL FIELDS
    # ========================================================

    pts2d = np.column_stack(
        [y_flow, x_flow]
    )

    interp_thk = RegularGridInterpolator(
        (y, x),
        thk,
        bounds_error=False,
        fill_value=np.nan,
    )

    interp_usurf = RegularGridInterpolator(
        (y, x),
        usurf,
        bounds_error=False,
        fill_value=np.nan,
    )

    interp_topg = RegularGridInterpolator(
        (y, x),
        topg,
        bounds_error=False,
        fill_value=np.nan,
    )

    thk_f = interp_thk(pts2d)
    usurf_f = interp_usurf(pts2d)
    topg_f = interp_topg(pts2d)

    # ========================================================
    # INTERPOLATE 3D MODEL FIELDS
    # ========================================================

    
    npts = len(x_flow)

    interp_E = RegularGridInterpolator(
        (z, y, x),
        E,
        bounds_error=False,
        fill_value=np.nan,
    )

    interp_Epmp = RegularGridInterpolator(
        (z, y, x),
        Epmp,
        bounds_error=False,
        fill_value=np.nan,
    )

    pts3d = np.column_stack(
        [
            np.repeat(z, npts),
            np.tile(y_flow, nz),
            np.tile(x_flow, nz),
        ]
    )

    E3d = interp_E(
        pts3d
    ).reshape(
        nz,
        npts,
    )

    Epmp3d = interp_Epmp(
        pts3d
    ).reshape(
        nz,
        npts,
    )

    # ========================================================
    # ICE TYPE
    # ========================================================

    ice_type = np.where(
        E3d >= Epmp3d,
        1,
        0,
    )

    # ========================================================
    # VERTICAL COORDINATE
    # ========================================================

    vert_spacing = 4

    zeta_edges = (
        np.arange(nz + 1)
        / nz
    )

    zeta_edges = (
        zeta_edges / vert_spacing
    ) * (
        1.0
        + (vert_spacing - 1.0)
        * zeta_edges
    )

    zeta_mid = 0.5 * (
        zeta_edges[:-1]
        + zeta_edges[1:]
    )

    # ========================================================
    # MODEL GEOMETRY
    # ========================================================

    X = np.tile(
        dist_km,
        (nz, 1),
    )

    Z = (
        topg_f[None, :]
        + zeta_mid[:, None]
        * (
            usurf_f - topg_f
        )[None, :]
    )

    # ========================================================
    # PLOT
    # ========================================================

    cmap_ice = ListedColormap(
        [
            "lightblue",
            "salmon",
        ]
    )

    norm_ice = BoundaryNorm(
        [-0.5, 0.5, 1.5],
        cmap_ice.N,
    )

    fig, ax = plt.subplots(
        figsize=(11, 5),
        dpi=150,
    )

    cf = ax.contourf(
        X,
        Z,
        ice_type,
        levels=[
            -0.5,
            0.5,
            1.5,
        ],
        cmap=cmap_ice,
        norm=norm_ice,
    )

    # ========================================================
    # OBSERVED CTS
    # ========================================================

    if plot_obs:

        valid_obs = (
            (cts_obs >= topg_f)
            & (cts_obs <= usurf_f)
            & (cts_obs > bed_obs)
        )

        cts_obs_plot = np.where(
            valid_obs,
            cts_obs,
            np.nan,
        )

        ax.plot(
            dist_km,
            cts_obs_plot,
            "k--",
            lw=2,
            label="Observed CTS",
        )

    # ========================================================
    # SURFACE
    # ========================================================

    ax.plot(
        dist_km,
        usurf_f,
        "k",
        lw=1.5,
        label="Surface",
    )

    # ========================================================
    # BED
    # ========================================================

    ax.plot(
        dist_km,
        topg_f,
        "k--",
        lw=1.2,
        label="Bed",
    )

    # ========================================================
    # ELA
    # ========================================================

    if plot_ela and hasattr(state, "ela"):

        ela = state.ela

        if hasattr(ela, "numpy"):
            ela = ela.numpy()

        ela = np.asarray(ela).squeeze()

        if np.ndim(ela) == 0:
            ela_value = ela.item()
        else:
            ela_value = ela.item()

        x_ela = np.argmin(
            np.abs(
                usurf_f - ela_value
            )
        )

        ax.plot(
            [
                dist_km[x_ela],
                dist_km[x_ela],
            ],
            [
                ela_value - 10,
                ela_value + 10,
            ],
            "k-",
            lw=2,
            label="ELA",
        )

    # ========================================================
    # BEDROCK
    # ========================================================

    z_min = np.nanmin(Z)

    ax.fill_between(
        dist_km,
        topg_f,
        z_min,
        color="saddlebrown",
        alpha=0.8,
        zorder=2,
        label="Bedrock",
    )

    # ========================================================
    # COLORBAR
    # ========================================================

    cbar = fig.colorbar(
        cf,
        ax=ax,
        ticks=[0, 1],
    )

    cbar.ax.set_yticklabels(
        [
            "Cold ice",
            "Temperate ice",
        ]
    )

    # ========================================================
    # LABELS
    # ========================================================

    ax.set_xlabel(
        "Distance along flowline (km)",
        fontsize=14,
    )

    ax.set_ylabel(
        "Altitude (m a.s.l.)",
        fontsize=14,
    )

    ax.tick_params(
        labelsize=14,
    )

    ax.set_title(
        f"Ice type vertical section – "
        f"{radar_line} – {time}",
        fontsize=16,
    )

    ax.grid(
        True,
        linestyle=":",
    )

    ax.legend()

    fig.tight_layout()

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    out_dir = os.path.join(
        ".",
        "Plots",
        "Ice_type",
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    # ========================================================
    # OUTPUT FILE
    # ========================================================

    fname = os.path.join(
        out_dir,
        f"{radar_line}_ice_type_{time}.png",
    )

    fig.savefig(
        fname,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)



    return fname


