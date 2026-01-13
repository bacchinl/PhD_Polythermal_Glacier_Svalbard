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
#from igm.modules.process.enthalpy.enthalpy import vertically_discretize_tf
# from analytical_solutions import validate_exp_a


# -------------------------------------------------------------------
# Configuration analytique
# -------------------------------------------------------------------
R = 1000.0  # rayon dome
A = 78 * 10 ** (-18)   # Arrhenius factor (MPa^-3 y^-1)
n = 3.0     # Glen exposant
rho = 910.0 # ice density (g/m^3)
g = 9.81  # earth's gravity (m/s^2)
k = A * (rho * g) ** n  # Generic constant (y^-1 m^-3)

# constant slope
theta = np.deg2rad(0.0)   # 0° de pente
S = np.sin(theta)

# Ice velocity
def analytical_velocity(H, z):
    """
    u(z) = 2 A (rho g S)^n / (n+1) * (H^(n+1) - (H - z)^(n+1))
    """
    coeff = 2.0 * A * (rho * g * S)**n / (n + 1.0)
    return coeff * (H**(n + 1.0) - (H - z)**(n + 1.0))


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
    cfg.processes.enthalpy.till.hydro.drainage_rate = 20.0
    cfg.processes.enthalpy.drainage.drain_ice_column = False
    cfg.processes.enthalpy.solver.allow_basal_refreezing = True
    cfg.processes.enthalpy.solver.correct_w_for_melt = False

    # define geometry
    x = np.arange(-100, 101) * 1000  # make x-axis, lenght 100 km,
    y = np.arange(-100, 101) * 1000  # make y-axis, lenght 100 km,
    Ny, Nx =len(y), len(x)

    X, Y = np.meshgrid(x, y)
    L = np.sqrt(X**2 + Y**2)

    topg = np.zeros_like(X) # no slope
    usurf = np.maximum(1500 * (1 - (L / 100000) ** 2), topg) 
    thk = usurf
    
    state = State()
    state.x = tf.constant(x.astype("float32"))
    state.y = tf.constant(y.astype("float32"))
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    state.dx = tf.constant(dx * tf.ones((Ny, Nx), dtype=tf.float32))
    state.dy = tf.constant(dy * tf.ones((Ny, Nx), dtype=tf.float32)) 
    
    state.dX = tf.Variable(dx * tf.ones((Ny, Nx), dtype=tf.float32))
    state.dY = tf.constant(dy * tf.ones((Ny, Nx), dtype=tf.float32))


    state.topg = tf.Variable(topg.astype("float32"))
    state.thk = tf.Variable(thk.astype("float32"))
    state.usurf = tf.Variable(usurf.astype("float32"))

    state.z, state.dz = vertical_discr(state.thk, Nz)


    # time
    state.t = tf.Variable(0.0, trainable=False)
    state.dt = tf.Variable(dt, trainable=False)
    
    # velocity
    state.U = tf.Variable(tf.zeros((Nz, Ny, Nx)), trainable=False)
    state.V = tf.Variable(tf.zeros((Nz, Ny, Nx)), trainable=False)
    state.W = tf.Variable(tf.zeros((Nz, Ny, Nx)), trainable=False)

        
    state.U.assign(analytical_velocity(state.thk[None, :, :], state.z))
    state.V.assign(analytical_velocity(state.thk[None, :, :], state.z))
    
    # others 
    state.air_temp = tf.Variable(-10.0 * tf.ones((1, Ny, Nx)), trainable=False)
    state.basal_heat_flux = 0.042 * tf.ones((Ny, Nx))


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
    time_simu = 100.0
    time_list = np.arange(0, time_simu, dt) + dt

    times = []
    T_store = []
    drainage_store = []
    U_store = []
    W_store = []
    E_store = []

    for it, t in enumerate(time_list):
        state.t.assign(float(t))

        enthalpy.update(cfg, state)

        times.append(float(t))
        T_store.append(state.T.numpy())
        drainage_store.append(state.h_water_till.numpy())
        U_store.append(state.U.numpy())
        W_store.append(state.W.numpy())
        E_store.append(state.E.numpy())

    return (
        np.array(times),
        np.array(T_store),
        np.array(drainage_store),
        np.array(U_store),
        np.array(W_store),
        np.array(E_store)
    )

    
cfg, state = _setup_experiment_Hewitt(dt=1.0)
times, T, drainage, U, W, E = _run_simu(cfg, state, dt=1.0)


def _plot_results(times, T, drainage, U, W, E, x, z):
    """ Plot the results to check, trying to copy Wang"""
    T_last = T[-1]      # (Nz,Ny,Nx)
    print("Mean T : ", np.mean(T_last)+273.15, "°C")

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
        cmap="turbo"
    )
    plt.colorbar(pcm, label="Temperature (°C)")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(f"Temperature field at t = {times[-1]:.1f} years")
    plt.tight_layout()
    plt.show()
    plt.savefig("Plots/temperature_ice_cap_{time_simu}.png")
    plt.close()

    ### ENTHALPY 

    E_last = E[-1]          
    u_last = U[-1]
    w_last = W[-1]

    E_section = E_last[:, iy, :] 
    ux = u_last[:, iy, :]      # horizontal velocity
    uz = w_last[:, iy, :]

    E_sub = E_section[:, mask]
    ux = ux[:, mask]
    uz = uz[:, mask]

    plt.figure(figsize=(9,5))

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
    Xs, Zs = np.meshgrid(x_sub, z_sub)

        
    #ax1.streamplot(
     #   Xs, Zs,
     #   ux, uz,
     #   color="white",
     #   linewidth=1,
     #   density=1.2,
     #   arrowsize=1.2
    #)

    ax1.set_xlim(x_sub.min(), x_sub.max())

    ax1.set_xlabel("x (km)")
    ax1.set_ylabel("z (m)")
    ax1.set_title(f"Enthalpy and velocities at t = {times[-1]:.1f} ans")
    
    ax2 = plt.subplot(2,1,2)
    
    drainage_last = drainage[-1]
    print(" ---- DRAINAGE ----", " Min : ", np.min(drainage_last), " Max : ", np.max(drainage_last), "N dims : ", drainage_last.ndim, "Drainage shape : ", drainage_last.shape)
    print(" ---- Axes ----", "X : ", x_sub.shape, "Z : ", z_sub.shape)
    drainage_profile = drainage_last[mask,:]

    
    ax2.plot(x_sub, drainage_profile, label="drainage")
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

_plot_results(times, T, drainage, U, W, E, x, z)
