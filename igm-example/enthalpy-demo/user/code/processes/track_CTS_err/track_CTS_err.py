#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""

Track observation of CTS elevation, compare with modelled one

==============================================================================

Input: ---
Output: ----
"""

# Import the most important libraries
import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from netCDF4 import Dataset
import glob
import pandas as pd


def initialize(cfg,state):
        

    #### OBS ###
    glacier_name = cfg.processes.track_CTS_err.glacier_name
    data_dir = os.path.expanduser(cfg.processes.track_CTS_err.data_dir)
    year_obs = cfg.processes.track_CTS_err.year_obs

    csv_pattern = os.path.join(
        data_dir,
        f"thickness_cts_points_{glacier_name}*.csv"
    )

    print("Répertoire courant :", os.getcwd())

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"The data directory does not exist: '{data_dir}'. "
            f"Please check 'processes.track_CTS_err.data_dir' in the configuration file."
        )


    csv_files = sorted(glob.glob(csv_pattern))
    print("Nb fichiers trouvés", len(csv_files))
    state.gpr_profiles = []

    for csv_file in csv_files:
        print("file trouvé")
        radar_line = (
            os.path.basename(csv_file)
            .replace("thickness_cts_points_", "")
            .replace(".csv", "")
        )
        columns_to_keep = ["radar_key", "easting", "northing", "temperate_elevation"]
        gpr_df = pd.read_csv(
                csv_file,
                usecols=columns_to_keep
        )
        #print(gpr_df["radar_key"])
        x_gpr = gpr_df["easting"].values
        y_gpr = gpr_df["northing"].values
        cts_obs = gpr_df["temperate_elevation"].values
        
        i_gpr = np.argmin(
            np.abs(state.x[None, :] - x_gpr[:, None]),
            axis=1,
        )

        j_gpr = np.argmin(
            np.abs(state.y[None, :] - y_gpr[:, None]),
            axis=1,
        )

        state.gpr_profiles.append(
            {
                "name": radar_line,
                "x": x_gpr,
                "y": y_gpr,
                "cts_obs": cts_obs,
                "i": i_gpr,
                "j": j_gpr,
            }
        )
    
def update(cfg,state):
    pass

def finalize(cfg,state):
    
    E = state.E.numpy()          # (Nz,Ny,Nx)
    E_pmp = state.E_pmp.numpy()
    
    Nz = E.shape[0]

    vert_spacing = 4

    zeta_edges = np.arange(Nz + 1) / Nz
    zeta_edges = (
        zeta_edges / vert_spacing
    ) * (1 + (vert_spacing - 1) * zeta_edges)

    zeta_mid = 0.5 * (
        zeta_edges[:-1] + zeta_edges[1:]
    )

    rmse_all = []
    #print("Nb profil", len(state.gpr_profiles))
    for profile in state.gpr_profiles:

        cts_model = np.full(len(profile["i"]), np.nan)
        #print(f"{profile['name']} : {len(profile['i'])} points")
        for p, (i, j) in enumerate(zip(profile["i"], profile["j"])):

            Ecol = E[:, j, i]
            Epcol = E_pmp[:, j, i]

            delta = Ecol - Epcol
            #print(np.min(delta), np.max(delta))
            ind = np.where(np.diff(np.sign(delta)) != 0)[0]
            #print(len(ind))
            if len(ind) == 0:
                continue

            k = ind[0]

            z = state.topg[j, i] + zeta_mid * state.thk[j, i]
    
            d1 = delta[k]
            d2 = delta[k + 1]

            z1 = z[k]
            z2 = z[k + 1]

            if np.abs(d2 - d1) < 1e-12:
                cts_model[p] = z1
            else:
                cts_model[p] = z1 - d1 * (z2 - z1) / (d2 - d1)

        mask = (
            np.isfinite(cts_model)
            & np.isfinite(profile["cts_obs"])
        )

        if np.any(mask):

            rmse = np.sqrt(
                np.mean(
                    (cts_model[mask] - profile["cts_obs"][mask]) ** 2
                )
            )

            rmse_all.append(rmse)

            print(
                f"{profile['name']} : RMSE CTS = {rmse:.2f} m"
            )

    if len(rmse_all) > 0:

        cost = float(np.mean(rmse_all))

    else:

        cost = 1e6

    print(f"\nMean CTS RMSE : {cost:.2f} m")

    if not hasattr(state, "score"):
        state.score = {}

    state.score["cost_cts"] = cost
    
