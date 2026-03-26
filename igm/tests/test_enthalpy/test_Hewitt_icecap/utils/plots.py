#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Plotting utilities for the Hewitt & Schoof (2017) ice cap enthalpy test."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Any
from scipy.interpolate import RectBivariateSpline

from igm.common import State


def plot_results(
    results: Dict[str, Any],
    state: State,
    output_dir: str = ".",
    T_s: float = -10.0,
    drain: bool = True,
) -> None:
    """
    Generate diagnostic plots for the Hewitt ice cap test.

    Produces two figures:
      - temperature_ice_cap_Ts<T_s>_<drain|nodrain>.png
      - enthalpy_ice_cap_Ts<T_s>_<drain|nodrain>.png
    """
    ts_tag = f"Ts{int(T_s)}".replace("-", "m")
    drain_tag = "drain" if drain else "nodrain"
    tag = f"{ts_tag}_{drain_tag}"
    x_km = state.x.numpy() / 1000.0
    times = results["times"]
    T = results["T"]  # (Nt, Nz, Ny, Nx)
    omega = results["omega"]  # (Nt, Nz, Ny, Nx)
    E = results["E"]
    E_pmp = results["E_pmp"]
    drainage = results["drainage"]  # (Nt, Ny, Nx)

    zeta = state.iceflow.discr_v.enthalpy.zeta.numpy()  # (Nz,)
    thk = state.thk.numpy()  # (Ny, Nx)
    z_all = zeta[:, np.newaxis, np.newaxis] * thk[np.newaxis, :, :]  # (Nz, Ny, Nx)

    iy = T.shape[2] // 2  # central y-slice
    mask = (x_km >= 0.0) & (x_km <= 100.0)
    x_sub = x_km[mask]

    # --- Figure 1: Temperature cross-section at final time + depth-averaged evolution ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    T_last = T[-1, :, iy, :] - 273.15  # (Nz, Nx), Celsius
    z_last = z_all[:, iy, :]  # (Nz, Nx)
    pcm = ax1.pcolormesh(
        x_sub,
        z_last[:, mask],
        T_last[:, mask],
        shading="auto",
        cmap="turbo",
        vmin=-10,
        vmax=0,
    )
    plt.colorbar(pcm, ax=ax1, label="Temperature (°C)")
    ax1.set_xlabel("x (km)")
    ax1.set_ylabel("z (m)")
    ax1.set_title(
        f"Temperature at t = {times[-1]:.0f} yr  (Hewitt & Schoof 2017, Fig. 7)"
    )

    dt_curve = times[-1] / 10.0
    requested = np.arange(0, times[-1] + 0.5 * dt_curve, dt_curve)
    for t_req in requested:
        i = int(np.argmin(np.abs(times - t_req)))
        T_mean = np.mean(T[i, :, iy, :] - 273.15, axis=0)  # depth-averaged, (Nx,)
        ax2.plot(x_sub, T_mean[mask], label=f"{times[i]:.0f} yr")

    ax2.set_xlabel("x (km)")
    ax2.set_ylabel("Depth-averaged temperature (°C)")
    ax2.set_title("Evolution of depth-averaged temperature")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/temperature_ice_cap_{tag}_high.png", dpi=150)
    plt.close()



    # --- Figure 2: Temperature / water fraction and basal drainage at final time ---
    

    E_last = E[-1, :, iy, :]
    Epmp_last = E_pmp[-1, :, iy, :]
    omega_last = omega[-1, :, iy, :]


    # Cold regions: show temperature; temperate regions: show water fraction
    cold = (E_last - Epmp_last) < 0.0
    field = np.zeros_like(T_last)
    field[cold] = T_last[cold]        # temperature
    field[~cold] = omega_last[~cold]*10  # water fraction




    
    field_plot = field[:, mask]
    z_plot = z_last[:, mask]

    
    # ---------------------------
    # Figure layout (Hewitt style)
    # ---------------------------

    fig = plt.figure(figsize=(10,5))
    gs = fig.add_gridspec(2,1,height_ratios=[4,1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    #plt.savefig(f"{output_dir}/hewitt_fig7_{tag}.png", dpi=200)
    # ---------------------------
    # Main section
    # ---------------------------

    # Nombre de couleurs pour chaque segment
    N_temp = 200  # turbo
    N_omega = 200 # blues

    # Récupérer les palettes
    cmap_temp = plt.cm.jet(np.linspace(0, 1, N_temp))
    cmap_omega = plt.cm.Blues(np.linspace(0, 1, N_omega))

    # Créer la palette combinée
    cmap_combined = np.vstack([cmap_temp, cmap_omega])
    cmap_field = mcolors.ListedColormap(cmap_combined)

    norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=3)

    X = np.tile(x_sub, (field_plot.shape[0], 1))      # (Nz, Nx_mask)
    Z = z_plot  
    

    cf = ax1.contourf(
    X,
    Z,
    field_plot,
    levels=200,
    cmap=cmap_field,
    norm = norm,
    vmin=-10,
    vmax=3
    )

    cb = fig.colorbar(cf, ax=[ax1, ax2], fraction=0.045, pad=0.04)
    cb.set_label("Temperature (°C) / Water fraction (x10)")
    cb.set_ticks([-10, -5, 0, 1, 2])

    ax1.set_ylabel("z (m)")
    ax1.set_title(f"Polythermal structure at t={times[-1]:.0f} yr")
    #plt.savefig(f"{output_dir}/hewitt_fig7_{tag}.png", dpi=200)
    # ---------------------------
    # Streamlines (white)
    # ---------------------------

    #if hasattr(state, "U"):
    streamlines = True
    if streamlines:
        
        U = state.U[:,iy,:].numpy()
        U = U[:,mask]
        Nx_sub = U.shape[1]
        x_stream = np.linspace(x_sub.min(), x_sub.max(), Nx_sub)

        Nz = U.shape[0]
        z_stream = np.linspace(0, np.max(z_plot), Nz)
        W = state.W[:,iy,:].numpy()
        Nz, Nx = U.shape 
 
        W = W[:,mask]
        U = np.nan_to_num(U)
        W = np.nan_to_num(W)
        x_i = np.linspace(x_stream.min(), x_stream.max(), 300)
        z_i = np.linspace(z_stream.min(), z_stream.max(), 300)

        
        fU = RectBivariateSpline(z_stream, x_stream, U)
        fW = RectBivariateSpline(z_stream, x_stream, W)

        U_i = fU(z_i, x_i)
        W_i = fW(z_i, x_i) 
    
        
        print("SHAPE OF U ", U.shape)
        print("W shape:", W.shape)
        print("x_sub:", x_sub.shape)
        print("z_plot:", z_plot.shape)
        
        Nz = U.shape[0]

        ax1.streamplot(x_i,z_i,U_i,W_i*1000,color="white",density=0.5,linewidth=1.5,arrowstyle='-')

    # ---------------------------
    # Drainage panel
    # ---------------------------

    drain_last = drainage[-1, iy, mask] * 1000.0

    ax2.plot(x_sub, drain_last, color="black", lw=2)

    ax2.set_ylabel("melt (mm yr⁻¹)")
    ax2.set_xlabel("x (km)")
    ax2.set_ylim(0, 25)
    ax2.invert_yaxis()

    ax2.grid(True, alpha=0.3)

    #plt.tight_layout()
    print("Saving figure to:", f"{output_dir}/hewitt_fig7_{tag}.png")
    plt.savefig(f"{output_dir}/hewitt_fig7_{tag}_high.png", dpi=200)
    print("Saving figure to:", f"{output_dir}/hewitt_fig7_{tag}.png")
    plt.close()

    
