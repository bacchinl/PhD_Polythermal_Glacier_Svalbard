#!/usr/bin/env python3
"""
Test ice caps experiment from Hewitt and Schoof (2017):
"Enthalpy benchmark experiments for numerical ice sheet models"
"""

import os
import numpy as np
import tensorflow as tf
import pytest
import matplotlib.pyplot as plt

import igm
from igm.common import State
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes.enthalpy import enthalpy
from igm.processes.iceflow import iceflow
from igm.processes.enthalpy.temperature.utils import compute_pmp_tf
#from igm.modules.process.enthalpy.enthalpy import vertically_discretize_tf
# from analytical_solutions import validate_exp_a


# -------------------------------------------------------------------
# Configuration analytique
# -------------------------------------------------------------------

A = 2.4 *10 **(-24) #  Pa-3 s-1  78 * 10 ** (-18)   # Arrhenius factor (MPa^-3 y^-1)
n = 3.0     # Glen exposant
rho = 910.0 # ice density (g/m^3)
g = 9.81  # earth's gravity (m/s^2)
k = A * (rho * g) ** n  # Generic constant (y^-1 m^-3)




def compute_icecap_velocity(state, A, rho, g, n, H0, R0):
    """
    Compute analytical 3D velocity field (U,V,W)
    for an axisymmetric Hewitt ice-cap configuration.

    Returns:
        U, V, W  with shape (Nz, Ny, Nx)
    """

    # ---------------------------
    # Geometry
    # ---------------------------
    x = tf.cast(state.x, tf.float32)
    y = tf.cast(state.y, tf.float32)

    X, Y = tf.meshgrid(x, y)
    X = tf.cast(X, tf.float32)
    Y = tf.cast(Y, tf.float32)

    R = tf.sqrt(X**2 + Y**2)
    R_safe = tf.where(R == 0.0, tf.ones_like(R), R)

    # Ice thickness H(R) = surface
    H = H0 * (1.0 - (R / R0) ** 2.0)
    H = tf.maximum(H, 0.0)
    H3 = H[None, :, :]     # -> (Nz,Ny,Nx)

    # ---------------------------
    # Vertical coordinate
    # ---------------------------
    z = tf.cast(state.z, tf.float32)    # (Nz,Ny,Nx)

    # ---------------------------
    # Surface slope magnitude
    # S = dZs/dR = 2 H0 R / R0^2
    # ---------------------------
    S = (2.0 * H0 / (R0**2)) * R_safe
    S3 = S[None, :, :]

    # ---------------------------
    # Speed magnitude (Glen law)
    # ---------------------------
    coef = 2.0 * A * (rho * g) ** n / (n + 1.0)
    Umag = coef * (S3**n) * (H3**(n + 1.0) - (H3 - z)**(n + 1.0))

    # ---------------------------
    # Horizontal direction (radial)
    # ---------------------------
    ex = X / R_safe
    ey = Y / R_safe

    ex3 = ex[None, :, :]
    ey3 = ey[None, :, :]

    U = Umag * ex3
    V = Umag * ey3

    # ---------------------------
    # Vertical velocity from incompressibility
    # div(u) + dW/dz = 0
    # ---------------------------
    dx = tf.cast(state.dx[0,0], tf.float32)
    dy = tf.cast(state.dy[0,0], tf.float32)

    dUdx = (U[:, :, 2:] - U[:, :, :-2]) / (2.0 * dx)
    dVdy = (V[:, 2:, :] - V[:, :-2, :]) / (2.0 * dy)

    # pad to original size
    dUdx = tf.pad(dUdx, [[0,0],[0,0],[1,1]], mode="SYMMETRIC")
    dVdy = tf.pad(dVdy, [[0,0],[1,1],[0,0]], mode="SYMMETRIC")

    div_h = dUdx + dVdy   # horizontal divergence

    # integrate vertically from base: W(z) = -∫ div dz
    dz = state.dz
    dz3 = dz

    W = -tf.cumsum(div_h * dz3, axis=0)

    return U, V, W



def vertical_discr(thk, Nz):
    """Create vertical coordinates dependent on ice thickness H(x,y)."""

    Ny, Nx = thk.shape

    # sigma levels centered in each layer: (Nz,)
    k = tf.range(Nz, dtype=tf.float32)
    sigma = (k + 0.5) / Nz

    # reshape to broadcast over (Ny,Nx)
    sigma = sigma[:, None, None]          # (Nz,1,1)

    # thickness as (1,Ny,Nx) to broadcast
    H = thk[None, :, :]                   # (1,Ny,Nx)

    # vertical coordinates: z from bed to surface
    z = sigma * H

    # layer thickness dz = H / Nz
    dz = tf.broadcast_to(H / Nz, z.shape)

    return z, dz



def compute_surface_gradients(usurf, dx, dy):
    """
    Compute slopes of a 2D surface using central differences.
    Vectorized version compatible with TensorFlow (no item assignment).
    """
    # shift left/right for x-gradient
    usurf_right = tf.concat([usurf[:, 1:], usurf[:, -1:]], axis=1)
    usurf_left = tf.concat([usurf[:, :1], usurf[:, :-1]], axis=1)
    slope_x = (usurf_right - usurf_left) / (2.0 * dx)

    # shift up/down for y-gradient
    usurf_up = tf.concat([usurf[1:, :], usurf[-1:, :]], axis=0)
    usurf_down = tf.concat([usurf[:1, :], usurf[:-1, :]], axis=0)
    slope_y = (usurf_up - usurf_down) / (2.0 * dy)

    return slope_x, slope_y


# -------------------------------------------------------------------
# INITIALIZATION
# -------------------------------------------------------------------

def _setup_experiment_Hewitt(dt):
    """Initialize configuration and state """
    # Load configuration
    cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))

    # Configure vertical discretization
    Nz = 50
    cfg.processes.iceflow.numerics.Nz = Nz
    cfg.processes.iceflow.numerics.vert_spacing = 1
    cfg.processes.enthalpy.numerics.Nz = Nz
    cfg.processes.enthalpy.numerics.vert_spacing = 1

    # Configure params for Hewitt experiment
    cfg.processes.enthalpy.thermal.K_ratio = 1.0e-5
    cfg.processes.enthalpy.till.hydro.drainage_rate = 40.0
    cfg.processes.enthalpy.drainage.drain_ice_column = True
    cfg.processes.enthalpy.solver.allow_basal_refreezing = False
    cfg.processes.enthalpy.solver.correct_w_for_melt = False

    # define geometry
    x = np.arange(-100, 101) * 1000  # make x-axis, lenght 100 km,
    y = np.arange(-2, 2) * 1000  # make y-axis, lenght 100 km,
    Ny, Nx =len(y), len(x)
    H0, R0 = 1500, 100000

    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    topg = np.zeros_like(X) # no slope
    usurf = np.maximum(H0 * (1 - (R / R0) ** 2), topg) 
    thk = usurf
    
    state = State()
    state.x = tf.constant(x.astype("float32"))
    state.y = tf.constant(y.astype("float32"))
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    state.dx = tf.constant(dx * tf.ones((Ny, Nx), dtype=tf.float32))
    state.dy = tf.constant(dy * tf.ones((Ny, Nx), dtype=tf.float32)) 
    
    state.dX = tf.Variable(dx * tf.ones((Ny, Nx), dtype=tf.float32))
    state.dY = tf.Variable(dy * tf.ones((Ny, Nx), dtype=tf.float32))


    state.topg = tf.Variable(topg.astype("float32"))
    state.thk = tf.Variable(thk.astype("float32"))
    state.usurf = tf.Variable(usurf.astype("float32"))

    state.z, state.dz = vertical_discr(state.thk, Nz)


    # time
    state.t = tf.Variable(0.0, trainable=False)
    state.dt = tf.Variable(dt, trainable=False)
    
    # velocity  initialization
    state.U = tf.Variable(tf.zeros((Nz, Ny, Nx)), trainable=False)
    state.V = tf.Variable(tf.zeros((Nz, Ny, Nx)), trainable=False)
    state.W = tf.Variable(tf.zeros((Nz, Ny, Nx)), trainable=False)

    U, V, W = compute_icecap_velocity(state, A, rho, g, n, H0, R0)

    state.U.assign(U)
    state.V.assign(V)
    state.W.assign(W)

    print(" ----INIT--------- Vals mean ux, uy, uz : ", np.mean(state.U), np.mean(state.V), np.mean(state.W))
    state.iceflow = (state.U, state.V, state.W)

    # others 
    state.air_temp = tf.Variable(-10.0 * tf.ones((1, Ny, Nx)), trainable=False)
    state.basal_heat_flux = 0.065 * tf.ones((Ny, Nx))


    # Initialize enthalpy module with temperate ice (0°C)
    #enthalpy.initialize(cfg, state)

    T_init = 262.15 * tf.ones((Nz, Ny, Nx)) # paper WANG T init =-11 deg cels
    state.E = cfg.processes.enthalpy.thermal.c_ice * (
        T_init - cfg.processes.enthalpy.thermal.T_ref
    )
    state.T = T_init
    state.omega = tf.zeros_like(state.E)
    state.h_water_till = tf.zeros((Ny, Nx)) # no water in the till at init
    
    enthalpy.initialize(cfg, state)
    return cfg, state


def _run_simu(cfg, state, dt):
    time_simu = 500.0
    time_list = np.arange(0, time_simu, dt) + dt

    times = []
    T_store = []
    drainage_store = []
    U_store = []
    W_store = []
    E_store = []
    SH_store = []
    omega_store =[]

    for it, t in enumerate(time_list):
        state.t.assign(float(t))

        #iceflow.update(cfg, state)
        enthalpy.update(cfg, state)
        #T_pmp, E_pmp = compute_pmp_tf(rho, g, depth_ice, beta, c_ice, T_pmp_ref, T_ref)

        times.append(float(t))
        T_store.append(state.T.numpy())
        drainage_store.append(state.basal_melt_rate.numpy())
        U_store.append(state.U.numpy())
        W_store.append(state.W.numpy())
        E_store.append(state.E.numpy())
        SH_store.append(state.strain_heat.numpy())
        omega_store.append(state.omega.numpy())

    return (
        np.array(times),
        np.array(T_store),
        np.array(drainage_store),
        np.array(U_store),
        np.array(W_store),
        np.array(E_store),
        np.array(SH_store),
        np.array(omega_store)
    )

    
cfg, state = _setup_experiment_Hewitt(dt=1.0)
times, T, drainage, U, W, E, strain_heat, omega = _run_simu(cfg, state, dt=1.0)


def _plot_results_T(times, T, drainage, U, W, E, strain_heat, omega, x, z):
    """ Plot the results to check, trying to copy Wang"""
    it = -1
    T_last = T[it]      # (Nz,Ny,Nx)
    print("  ---    Mean T      : ", np.mean(T_last)-273.15, "°C")

    Nz, Ny, Nx = T_last.shape
    # selecte the section y=0
    iy = Ny // 2
    T_section = T_last[:, iy, :] - 273.15
    
    x_km = x / 1000.0
    # extract z
    if z.ndim == 3:
        z_section = z[:, iy, :]      # (Nz, Nx)
    else:
        # si z = (Nz,1,1)
        z_section = np.broadcast_to(z, (Nz, Ny, Nx))[:, iy, :]

    # PLot fig
    mask = (x_km >= 0) & (x_km <= 100)    # m
    x_sub = x_km[mask]
    T_sub = T_section[:, mask]
    z_sub = z_section[:, mask]
    
    X, Z = np.meshgrid(x_sub, np.arange(Nz))

    plt.figure(figsize=(9,5))
    
    
    pcm = plt.pcolormesh(
        x_sub,
        z_sub,          # niveaux z
        T_sub,
        shading="auto",
        cmap="turbo", vmin=-10, vmax=0
    )
    plt.colorbar(pcm, label="Temperature (°C)")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(f"Temperature field at t = {times[-1]:.1f} years")
    plt.tight_layout()
    plt.show()
    plt.savefig("Plots/temperature_ice_cap_{time_simu}.png")
    plt.close()



    plt.figure(figsize=(9,5))
    dt_curve =len(times)/10

    # liste des années demandées
    requested = np.arange(0, times[-1]+1, dt_curve)

    for t_req in requested:
        # trouver snapshot le plus proche
        i = np.argmin(np.abs(times - t_req))

        # coupe y = 0
        T_section = T[i, :, iy, :] - 273.15     # (Nz, Nx)

        # moyenne verticale
        T_mean_z = np.mean(T_section, axis=0)   # (Nx,)
        T_mean_z = T_mean_z[mask]

        plt.plot(x_sub, T_mean_z, label=f"{times[i]:.0f} yr")

    plt.xlabel("x (km)")
    plt.ylabel("Mean Temperature over depth (°C)")
    plt.title("Evolution of depth-averaged temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()


    
def _plot_results_E(times, T, drainage, U, W, E, strain_heat, omega, x, z):
    

    """ Plot the results to check, trying to copy Wang"""
    it = -1
    T_last = T[it]      # (Nz,Ny,Nx)
    print("  ---    Mean T      : ", np.mean(T_last)-273.15, "°C")

    Nz, Ny, Nx = T_last.shape
    # selecte the section y=0
    iy = Ny // 2
    T_section = T_last[:, iy, :] - 273.15

    x_km = x / 1000.0
    # extract z
    if z.ndim == 3:
        z_section = z[:, iy, :]      # (Nz, Nx)
    else:
        # si z = (Nz,1,1)
        z_section = np.broadcast_to(z, (Nz, Ny, Nx))[:, iy, :]

    # PLot fig
    mask = (x_km >= 0) & (x_km <= 100)    # m
    x_sub = x_km[mask]
    T_sub = T_section[:, mask]
    z_sub = z_section[:, mask]

    X, Z = np.meshgrid(x_sub, np.arange(Nz))

    ### ENTHALPY 

    E_last = E[it]
    #Epmp_last = Epmp[it]
    u_last = U[it]
    w_last = W[it]
    omega_last = omega[it]

    E_section = E_last[:, iy, :]
    #Epmp_section = Epmp_last[:, iy, :]
    omega_section = omega_last[:, iy, :]
    ux = u_last[:, iy, :]      # horizontal velocity
    uz = w_last[:, iy, :]

    E_sub = E_section[:, mask]
    #Epmp_sub = Epmp_section[:, mask]
    omega_sub = omega_section[:, mask]
    ux_sub = ux[:, mask]
    uz_sub = uz[:, mask]

    plt.figure(figsize=(10,10))

    ax1 = plt.subplot(2,1,1)

    pcm = ax1.pcolormesh(
        x_sub,
        z_sub,
        E_sub,
        shading="auto",
        cmap="turbo"
    )
    cb = plt.colorbar(pcm)
    cb.set_label("Enthalpy")

    # -------- STREAMLINES ----------
    # grille pour streamplot
    print(" ------------- Dims x_sub, z_sub : ", x_sub.shape, z_sub.shape)
    Xs, Zs = np.meshgrid(x_sub, z_sub[:,1])
    print(" ------------- Dims Xs, Zs : ", Xs.shape, Zs.shape)
    print(" ------------- Dims ux, Uz : ", ux_sub.shape, uz_sub.shape)
    
    print(" ------------- Vals mean ux, Uz : ", np.mean(ux_sub), np.mean(uz_sub))
    print(" -------- MIN MAX uz : ", np.min(uz_sub), np.max(uz_sub))
    print(" -------- omega max : ",  np.max(omega_sub))

    ax1.streamplot(
        Xs, Zs,
        ux_sub/1000, uz_sub,
        color="white",
        linewidth=1.5,
        density=0.6,
        arrowsize=1.2
    )

    ax1.set_xlim(x_sub.min(), x_sub.max())

    ax1.set_xlabel("x (km)")
    ax1.set_ylabel("z (m)")
    ax1.set_title(f"Enthalpy and velocities at t = {times[-1]:.1f} ans")
    
    ax2 = plt.subplot(2,1,2)
    
    drainage_last = drainage[it]
    print(" ---- DRAINAGE ----", " Min : ", np.min(drainage_last), " Max : ", np.max(drainage_last), "N dims : ", drainage_last.ndim, "Drainage shape : ", drainage_last.shape)
    print(" ---- Axes ----", "X : ", x_sub.shape, "Z : ", z_sub.shape)
    
    drainage_profile = drainage_last[iy,mask] *365*24*3600*1000 #mm/yr
    print(" ---- DRAINAGE profile----", " Min : ", np.min(drainage_profile), " Max : ", np.max(drainage_profile), "N dims : ", drainage_profile.ndim, "Drainage shape : ", drainage_profile.shape)

    
    ax2.plot(x_sub, drainage_profile, label="water flux from ice at bed (mm/yr)")
    ax2.set_xlim(x_sub.min(), x_sub.max())
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("Drainage")
    ax2.set_title("Water drainage vs x")
    ax2.grid(True)


    plt.tight_layout()
    plt.show()
    plt.savefig("Plots/Enthalpy_velocities_ice_cap_{time_simu}.png")




x = state.x.numpy()
z = state.z.numpy()

_plot_results_T(times, T, drainage, U, W, E, strain_heat, omega, x, z)
_plot_results_E(times, T, drainage, U, W, E, strain_heat, omega, x, z)
