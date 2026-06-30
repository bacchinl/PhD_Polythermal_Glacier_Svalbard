#!/usr/bin/env python3
"""
Test ice caps experiment from stroglaciaren:
"Mathematical modeling and numerical simulation of  polythermal glaciers" Aschwanden
& "A two-dimensional, higher-order, enthalpy-based thermomechanical ice flow model for mountain glaciers and its benchmark experiments" Wang
"""

import os
import numpy as np
import tensorflow as tf
import pytest
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import igm
from igm.common import State
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes.enthalpy import enthalpy
from igm.processes.iceflow import iceflow
from igm.processes.enthalpy.temperature.utils import compute_pmp_tf
from igm.processes.iceflow.vertical import VerticalDiscrs
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


x_res =50
z_res = 20



# ---------------------------
# Load data
# ---------------------------
data_path = "/Inputs"


vb_csv   = np.loadtxt(os.path.join(data_path, "Basal_vel_flowline.csv"), delimiter=",",skiprows=1)
vs_csv   = np.loadtxt(os.path.join(data_path, "Surf_vel_flowline.csv"), delimiter=",",skiprows=1)
T_csv = np.loadtxt(os.path.join(data_path, "Temperature_Stor.csv"), delimiter=",",skiprows=1)
#CTS_csv = np.loadtxt(os.path.join(data_path, "CTS_Stor_obs.csv"), delimiter=",",skiprows=1) # observed CTS in aschwanden et al
#CTS_csv = np.loadtxt(os.path.join(data_path, "CTS_Stor_mod.csv"), delimiter=",",skiprows=1) # modelled CTS in aschwanden et al
CTS_csv = np.loadtxt(os.path.join(data_path, "CTS_Stor_mod_IGM.csv"), delimiter=",",skiprows=1) # modelled CTS in IGM (manual picking)

bed_csv = np.loadtxt(os.path.join(data_path, "bed_Stor.csv"), delimiter=",",skiprows=1)
surf_csv = np.loadtxt(os.path.join(data_path, "Surf_Stor.csv"), delimiter=",",skiprows=1)


x_bed  = bed_csv[:, 0]
bed_1d = bed_csv[:, 1]

x_surf  = surf_csv[:, 0]
surf_1d = surf_csv[:, 1]

x_vb  = vb_csv[:, 0]
vb_1d = vb_csv[:, 1]

x_vs  = vs_csv[:, 0]
vs_1d = vs_csv[:, 1]

x_T = T_csv[:, 0]
T_1d = T_csv[:, 1]

x_CTS = CTS_csv[:, 0]
CTS_1d = CTS_csv[:, 1]


def compute_velocity(state, vb, vs, m=1.0):
    """
    Pseudo-3D velocity field using MOLHO vertical interpolation.

    Inputs:
        state : contains x, y, z, dx, dy, dz, thk
        vb    : basal velocity (Ny, Nx)
        vs    : surface velocity (Ny, Nx)
        m     : MOLHO exponent

    Returns:
        U, V, W  (Nz, Ny, Nx)
    """

    # ---------------------------
    # Geometry
    # ---------------------------
    H = tf.maximum(state.thk, 0.0)      # (Ny,Nx)
    H3 = H[None, :, :]                  # (1,Ny,Nx)

    z = tf.cast(state.z, tf.float32)    # (Nz,Ny,Nx)
    z_rel = z - state.topg[None, :, :] ## relative height
    z_norm = tf.clip_by_value(z_rel / H3, 0.0, 1.0) ## aka sigma

    # ---------------------------
    # MOLHO vertical interpolation
    # ---------------------------
    Umag = vb[None, :, :] + ((vs - vb)[None, :, :] * (1-(1-z_norm)**m))

    # ---------------------------
    # Flow direction (x only → flowline)
    # ---------------------------

    U = Umag
    V = tf.zeros_like(U)

    # ---------------------------
    # Vertical velocity (incompressibility)
    # ---------------------------
    dx = tf.cast(state.dx[0, 0], tf.float32)

    dUdx = (U[:, :, 2:] - U[:, :, :-2]) / (2.0 * dx)
    dUdx = tf.pad(dUdx, [[0,0],[0,0],[1,1]], mode="SYMMETRIC")

    div_h = dUdx

    dz = state.dz
    W = -tf.cumsum(div_h * dz, axis=0)

    return U, V, W




def vertical_discr(topg, usurf, Nz, vert_spacing=1.0):

    Ny, Nx = topg.shape

    k = tf.range(Nz, dtype=tf.float32)
    s = (k + 0.5) / Nz  # sigma uniform base
    
    # grid deformation with vert spacing
    if vert_spacing != 1.0:
        s = s ** vert_spacing

        # renormalisation pour garder s ∈ [0,1]
        s = s / tf.reduce_max(s)

    sigma = s[:, None, None]

    H = (usurf - topg)[None, :, :]

    z = topg[None, :, :] + sigma * H
    print(" SIGMA : ", z[:,0,10])

    dz = z[1:] - z[:-1]
    dz = tf.concat([dz[:1], dz], axis=0)

    return z, dz




def compute_surface_gradients(usurf, dx, dy):
    """
    Compute slopes of a 2D surface using central differences.
    Vectorized version compatible with TensorFlow (no item assignment),
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

def _setup_experiment_Stor(dt):
    """Initialize configuration and state """
    # Load configuration
    cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))
    frac_refr = 0.015

    # Configure vertical discretization
    Nz = 40
    vert_spacing = 1 
    cfg.processes.iceflow.numerics.Nz = Nz
    cfg.processes.iceflow.numerics.vert_spacing = vert_spacing
    cfg.processes.iceflow.physics.init_arrhenius =  70
    cfg.processes.enthalpy.numerics.Nz = Nz
    cfg.processes.enthalpy.numerics.vert_spacing = vert_spacing

    # Configure params for Hewitt experiment
    cfg.processes.enthalpy.thermal.K_ratio = 1.0e-5
    cfg.processes.enthalpy.thermal.T_ref = 273.15 ## which is T_ref (T0) for Aschwanden
    cfg.processes.enthalpy.till.hydro.drainage_rate = 0.001
    cfg.processes.enthalpy.drainage.drain_ice_column = True
    cfg.processes.enthalpy.solver.allow_basal_refreezing = True
    cfg.processes.enthalpy.solver.correct_w_for_melt = False
    cfg.processes.enthalpy.till.friction.u_ref = 50.0


    
    T_pmp_ref = cfg.processes.enthalpy.thermal.T_pmp_ref
    beta = cfg.processes.enthalpy.thermal.beta

    # define geometry
    x = np.arange(0, 3450, x_res)  # make x-axis,
    
    y = np.arange(-2, 2)   # make y-axis,
    Ny, Nx =len(y), len(x)


    X, Y = np.meshgrid(x, y)

    # interpolation flowline → grille modèle
    bed_interp = np.interp(x, x_bed, bed_1d)
    surf_interp = np.interp(x, x_surf, surf_1d)    
    T_interp = np.interp(x, x_T, T_1d)
    

    topg = np.repeat(bed_interp[None, :], Ny, axis=0)
    usurf = np.repeat(surf_interp[None, :], Ny, axis=0)
    T_air = tf.ones((1, Ny, 1), dtype=tf.float32) * T_interp[None, None, :] - 273.15
    

    thk = np.maximum(usurf - topg, 0.0)
    
    vb_interp = np.interp(x, x_vb, vb_1d)
    vs_interp = np.interp(x, x_vs, vs_1d)

    # extension pseudo-3D (constant in y)
    vb_2d = np.repeat(vb_interp[None, :], Ny, axis=0)
    vs_2d = np.repeat(vs_interp[None, :], Ny, axis=0)

    vb_tf = tf.constant(vb_2d, dtype=tf.float32)
    vs_tf = tf.constant(vs_2d, dtype=tf.float32)


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

    state.z, state.dz = vertical_discr(state.topg, state.usurf,  Nz, vert_spacing)


    # time
    state.t = tf.Variable(0.0, trainable=False)
    state.dt = tf.Variable(dt, trainable=False)
    
    # velocity  initialization
    state.U = tf.Variable(tf.zeros((Nz, Ny, Nx)), trainable=False)
    state.V = tf.Variable(tf.zeros((Nz, Ny, Nx)), trainable=False)
    state.W = tf.Variable(tf.zeros((Nz, Ny, Nx)), trainable=False)

    U, V, W = compute_velocity(state, vb_tf, vs_tf, m=3.0) 

    state.U.assign(U)
    state.V.assign(V)
    state.W.assign(W)

    
    # others 
    state.air_temp = tf.Variable(T_air) 
    state.basal_heat_flux = 0.042 * tf.ones((Ny, Nx)) # 0.042 TAble.1 ashwanden
    
    # Initialize enthalpy module with temperate ice (0°C)
    T_init = 273.15 * tf.ones((Nz, Ny, Nx), dtype=tf.float32)


    state.E = cfg.processes.enthalpy.thermal.c_ice * (
        T_init - cfg.processes.enthalpy.thermal.T_ref
    )
    state.T = T_init
    state.omega = tf.zeros_like(state.E) # omega initialized at 0 (aschwanden)
    state.h_water_till = tf.zeros((Ny, Nx)) # no water in the till at init
    
    enthalpy.initialize(cfg, state)
    state.iceflow.U = state.U
    state.iceflow.V = state.V
    state.iceflow.W = state.W

    return cfg, state


def _run_simu(cfg, state, dt):
    time_simu = 1000.0
    time_list = np.arange(0, time_simu, dt) + dt

    times = []
    T_store = []
    drainage_store = []
    U_store = []
    W_store = []
    E_store = []
    E_pmp_store = []
    SH_store = []
    omega_store =[]

    for it, t in enumerate(time_list):

        state.t.assign(float(t))

        enthalpy.update(cfg, state)
        
        times.append(float(t))
        T_store.append(state.T.numpy())
        drainage_store.append(state.basal_melt_rate.numpy())
        U_store.append(state.U.numpy())
        W_store.append(state.W.numpy())
        E_store.append(state.E.numpy())
        E_pmp_store.append(state.E_pmp.numpy())
        SH_store.append(state.strain_heat.numpy())
        omega_store.append(state.omega.numpy())

    return (
        np.array(times),
        np.array(T_store),
        np.array(drainage_store),
        np.array(U_store),
        np.array(W_store),
        np.array(E_store),
        np.array(E_pmp_store),
        np.array(SH_store),
        np.array(omega_store)
    )

    

cfg, state = _setup_experiment_Stor(dt=1.0)
times, T, drainage, U, W, E, E_pmp, strain_heat, omega = _run_simu(cfg, state, dt=1.0)


def _plot_results_T(times, topg, usurf, T, drainage, U, W, E, strain_heat, omega, x, z, CTS):
    """ Plot the results to check, trying to copy Wang"""
    it = -1
    T_last = T[it]      # (Nz,Ny,Nx)
    

    Nz, Ny, Nx = T_last.shape
    # selecte the section y=0
    iy = 0 #Ny // 2
    print("   ------  iy ", iy)
    T_section = T_last[:, iy, :] - 273.15
    usurf_section = usurf[iy, :]
    topg_section = topg[iy, :]

    ### matching palette to hewitt fig 5
    colors_T = plt.cm.Blues_r(np.linspace(0.2, 1, 7))
    cmap_T = ListedColormap(colors_T)
    bounds_T = [-6, -5, -4, -3, -2, -1, 0, 1]
    norm_T = BoundaryNorm(bounds_T, cmap_T.N)



    x_km = x 
    # extract z
    if z.ndim == 3:
        z_section = z[:, iy, :]      # (Nz, Nx)
    else:
        # si z = (Nz,1,1)
        z_section = np.broadcast_to(z, (Nz, Ny, Nx))[:, iy, :]

    # PLot fig
    mask = (x_km >= 0) & (x_km <= 5000)    # m
    x_sub = x_km[mask]
    T_sub = T_section[:, mask]
    z_sub = z_section[:, mask]
    

    H = usurf_section - topg_section   # (Nx,)
    X, _ = np.meshgrid(x_sub, np.arange(Nz))   # (Nz, Nx)
    

    plt.figure(figsize=(9,5))
    
    levels = np.linspace(-6, 0, 7)
    cont = plt.contourf(X, z_sub, T_sub, levels=levels, cmap=cmap_T, vmin=-6, vmax=1)
    cbar = plt.colorbar(cont, label="Temperature (]C)",ticks=[-6,-5, -4, -3, -2, -1, 0])

    # Isothermes + surfaces
    plt.contour(X, z_sub, T_sub, levels=10, colors="k", linewidths=0.3, alpha=0.4)
    plt.plot(x_sub, usurf_section, color="k", linewidth=1.5, label="Altitude")
    plt.plot(x_sub, topg_section, color="k", linewidth=1.5, label="Bedrock")
    plt.plot(CTS[0], CTS[1], color="k", linewidth=1.5, linestyle="--", label="CTS")
    
    plt.xlabel("Distance from Bergschrund (m)", fontsize =16)
    plt.ylabel("Elevation (m a.s.l)", fontsize = 16)
    plt.title(f"Temperature field at t = {times[-1]:.1f} years")
    plt.tight_layout()
    plt.savefig(f"Plots/Temperature_storglaciaren_{times[-1]:.1f}_year.png")
    plt.show()
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

    plt.xlabel("x (m)")
    plt.ylabel("Mean Temperature over depth (°C)")
    plt.title("Evolution of depth-averaged temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()


    
def _plot_results_E(times, T, drainage, U, W, E, Epmp, strain_heat, omega, x, z):
    """ Plot the results to check, trying to copy Wang"""
    it = -1
    E_last = E[it]      # (Nz,Ny,Nx)
    E_pmp_last= E_pmp[it] 

    print("  ---    Mean E      : ", np.mean(E_last), "J/kg, shape ",E_last.shape )

    Nz, Ny, Nx = E_last.shape
    # selecte the section y=0
    iy = 0#Ny // 2
    print("   ------  iy ", iy)
    E_section = E_last[:, iy, :] 
    E_pmp_section = E_pmp_last[:, iy, :] 
    usurf_section = usurf[iy, :]
    topg_section = topg[iy, :]

    x_km = x 
    # extract z
    if z.ndim == 3:
        z_section = z[:, iy, :]      # (Nz, Nx)
    else:
        # si z = (Nz,1,1)
        z_section = np.broadcast_to(z, (Nz, Ny, Nx))[:, iy, :]

    # PLot fig
    mask = (x_km >= 0) & (x_km <= 5000)    # m
    x_sub = x_km[mask]
    E_sub = E_section[:, mask]
    E_pmp_sub = E_pmp_section[:, mask]
    z_sub = z_section[:, mask]
    
    
    H = usurf_section - topg_section   # (Nx,)
    X, _ = np.meshgrid(x_sub, np.arange(Nz))   # (Nz, Nx)
    
    


    plt.figure(figsize=(9,5))
    
    levels = np.linspace(-12000, 5000, 18)
    cont = plt.contourf(X, z_sub, E_sub, levels=levels, cmap="RdYlBu_r", vmin=-12000, vmax=5000)
    cbar = plt.colorbar(cont, label="Enthalpy", ticks=[-10000,-8000, -6000, -4000, -2000, 0, 2000, 4000])

    # Isothermes + surfaces
    plt.contour(X, z_sub, E_sub, levels=10, colors="k", linewidths=0.3, alpha=0.4)
    plt.plot(x_sub, usurf_section, color="k", linewidth=1.5, label="Altitude")
    plt.plot(x_sub, topg_section, color="k", linewidth=1.5, label="Bedrock")
    plt.plot(CTS[0], CTS[1], color="k", linewidth=1.5, linestyle="--", label="CTS")
    
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(f"Enthalpy field at t = {times[-1]:.1f} years")
    plt.tight_layout()
    #plt.grid(True)
    plt.savefig(f"Plots/Enthalpy_storglaciaren_{times[-1]:.1f}_year.png")
    plt.show()
    plt.close()




    diff_E = E_sub-E_pmp_sub
    ice_type = np.zeros_like(E_sub)
    ice_type[E_sub >= E_pmp_sub] = 1
     
    cts = (E_sub >= E_pmp_sub).astype(float)

    cmap_ice = ListedColormap(["lightblue", "salmon"])

    


    plt.figure(figsize=(9,5))

    levels = 40
    cont = plt.contourf(X, z_sub, ice_type, levels=levels, cmap=cmap_ice)
    
    cbar = plt.colorbar(cont, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Cold ice", "Tempered ice"])

    # Isothermes + surfaces
    #plt.contour(X, z_sub, E_sub, levels=10, colors="k", linewidths=0.3, alpha=0.4)
    plt.plot(x_sub, usurf_section, color="k", linewidth=1.5, label="Altitude")
    plt.plot(x_sub, topg_section, color="k", linewidth=1.5, label="Bedrock")
    plt.plot(CTS[0], CTS[1], color="k", linewidth=1.5, linestyle="--", label="CTS")

    plt.xlabel("Distance from Bergschrund (m)", fontsize =16)
    plt.ylabel("Elevation (m a.s.l)", fontsize = 16)
    plt.title(f"Hydrothermal structure t = {times[-1]:.1f} years")
    plt.tight_layout()
    plt.savefig(f"Plots/Ice_type_storglaciaren_{times[-1]:.1f}_year.png")
    plt.show()
    plt.close()


    plt.figure(figsize=(9,5))
    dt_curve =len(times)/20

    # liste des années demandées
    requested = np.arange(0, times[-1]+1, dt_curve)

    for t_req in requested:
        # trouver snapshot le plus proche
        i = np.argmin(np.abs(times - t_req))

        # coupe y = 0
        E_section = E[i, :, iy, :]      # (Nz, Nx)

        # moyenne verticale
        E_mean_z = np.mean(E_section, axis=0)   # (Nx,)
        E_mean_z = E_mean_z[mask]

        plt.plot(x_sub, E_mean_z, label=f"{times[i]:.0f} yr")

    plt.xlabel("x (km)")
    plt.ylabel("Mean enthalpy over depth (°C)")
    plt.title("Evolution of depth-averaged enthalpy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()





def _plot_results_omega(times, T, drainage, U, W, E, Epmp, strain_heat, omega, x, z, CTS):
    """ Plot the results to check, trying to copy Wang"""
    it = -1
    w_last = omega[it]      # (Nz,Ny,Nx)
    

    Nz, Ny, Nx = w_last.shape
    
    # selecte the section y=0
    iy = 0#Ny // 2
    print("   ------  iy ", iy)
    w_section = w_last[:, iy, :] 
    
    usurf_section = usurf[iy, :]
    topg_section = topg[iy, :]

    x_km = x 
    # extract z
    if z.ndim == 3:
        z_section = z[:, iy, :]      # (Nz, Nx)
    else:
        # si z = (Nz,1,1)
        z_section = np.broadcast_to(z, (Nz, Ny, Nx))[:, iy, :]

    # PLot fig
    mask = (x_km >= 0) & (x_km <= 5000)    # m
    x_sub = x_km[mask]
    w_sub = w_section[:, mask]
    z_sub = z_section[:, mask]



    X, _ = np.meshgrid(x_sub, np.arange(Nz))   # (Nz, Nx)
    

    plt.figure(figsize=(9,5))
    
    levels = np.linspace(0, 16, 9)
    cont = plt.contourf(X, z_sub, 1000*w_sub, levels=levels, cmap="Reds")#, vmin=0, vmax=10) # times 1000 fo g water per kilo
    cbar = plt.colorbar(cont, label="Water fraction, $g_w~kg^-1$")

    # Isothermes + surfaces
    plt.contour(X, z_sub, 1000*w_sub, levels=10, colors="k", linewidths=0.3, alpha=0.4)
    plt.plot(x_sub, usurf_section, color="k", linewidth=1.5, label="Altitude")
    plt.plot(x_sub, topg_section, color="k", linewidth=1.5, label="Bedrock")
    plt.plot(CTS[0], CTS[1], color="k", linewidth=1.5, linestyle="--", label="CTS")
    
    plt.xlabel("Distance from Bergschrund (m)", fontsize =16)
    plt.ylabel("Elevation (m a.s.l)", fontsize = 16)
    plt.title(f"Water content fraction field at t = {times[-1]:.1f} years")
    plt.tight_layout()
    plt.savefig(f"Plots/water_storglaciaren_{times[-1]:.1f}_year.png")
    plt.show()
    plt.close()


    plt.figure(figsize=(9,5))
    dt_curve =len(times)/20

    # liste des années demandées
    requested = np.arange(0, times[-1]+1, dt_curve)

    for t_req in requested:
        # trouver snapshot le plus proche
        i = np.argmin(np.abs(times - t_req))

        # coupe y = 0
        E_section = E[i, :, iy, :]      # (Nz, Nx)

        # moyenne verticale
        E_mean_z = np.mean(E_section, axis=0)   # (Nx,)
        E_mean_z = E_mean_z[mask]

        plt.plot(x_sub, E_mean_z, label=f"{times[i]:.0f} yr")

    plt.xlabel("x (km)")
    plt.ylabel("Mean water content over depth")
    plt.title("Evolution of depth-averaged water content")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()



def plot_profile_vitesse(U, z, topg, usurf, ix, iy, component="u"):
    """
    Plot vertical velocity profile at one (ix, iy)

    Parameters
    ----------
    U : ndarray (Nz, Ny, Nx)
        Velocity component (U, V or W)
    z : ndarray
        Vertical coordinate (Nz,) or (Nz,Ny,Nx)
    topg : ndarray (Ny, Nx)
        Bedrock elevation
    usurf : ndarray (Ny, Nx)
        Surface elevation
    ix, iy : int
        Horizontal indices
    component : str
        Label for velocity component
    """

    Nz = U.shape[0]

    # --- extract vertical axis (sigma)
    if z.ndim == 1:
        zhat = z
    elif z.ndim == 3:
        zhat = z[:, iy, ix]
    else:
        raise ValueError("z must be 1D or 3D")
    print("---------- ZHAT : ", zhat)

    # --- physical elevation
    H = usurf[iy, ix] - topg[iy, ix]
    #z_norm = tf.clip_by_value(z / H[:,iy, ix], 0.0, 1.0) 
    z_abs = topg[iy, ix] + zhat #* H
    
    # --- velocity profile
    print("----- U shape : ", U.shape)
    print("              : ", U[-1,:, iy, ix])
    U_prof = U[-1,:, iy, ix]

    # --- plot
    plt.figure(figsize=(4,6))
    plt.plot(U_prof, z_abs, "-o", lw=2)
    plt.axhline(usurf[iy, ix], color="c", lw=1)
    plt.axhline(topg[iy, ix], color="r", lw=1, ls="--")

    plt.xlabel(f"{component} velocity (m/yr)")
    plt.ylabel("Elevation (m)")
    plt.title(f"Vertical profile at (ix={ix}, iy={iy})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_vel(state):
    it = -1

    x=state.x
    z=state.z

    U = state.U
    W = state.W

    print(" SHAPE U /", U.shape)

    usurf=state.usurf
    topg=state.topg

    



    vel = np.sqrt(U**2 + W**2)
    Nz, Ny, Nx = state.E.shape

    vel_section = vel[:,0,:]
    usurf_section = usurf[iy, :]
    topg_section = topg[iy, :]

    z_section = z[:, iy, :]
    X, _ = np.meshgrid(x, np.arange(Nz))



    plt.figure(figsize=(9,5))

    levels = np.linspace(0, 28, 29)
    cont = plt.contourf(X, z_section, vel_section, levels=levels, cmap="jet", vmin=0, vmax=28)
    cbar = plt.colorbar(cont, label="Velocity ($m~yr^{-1}$)")

     # Isothermes + surfaces
    plt.contour(X, z_section, vel_section, levels=10, colors="k", linewidths=0.3, alpha=0.4)
    plt.plot(x, usurf_section, color="k", linewidth=1.5, label="Altitude")
    plt.plot(x, topg_section, color="k", linewidth=1.5, label="Bedrock")
    
    plt.xlabel("Distance from Bergschrund (m)", fontsize =16)
    plt.ylabel("Elevation (m a.s.l)", fontsize = 16)
    plt.title(f"Velocity field")
    plt.savefig(f"Plots/Velocity_Storglaciaren.png")
    plt.show()



def plot_CTS(state):
    topg = state.topg
    usurf =state.usurf
    x = state.x
    z= state.z[:, iy, :]

    usurf = usurf[0, :]
    topg = topg[0, :]
    
    Nz, Ny, Nx = state.E.shape
    
    CTS_obs = np.loadtxt(os.path.join(data_path, "CTS_Stor_obs.csv"), delimiter=",",skiprows=1)
    CTS_igm = np.loadtxt(os.path.join(data_path, "CTS_Stor_mod_IGM.csv"), delimiter=",",skiprows=1)
    CTS_wng = np.loadtxt(os.path.join(data_path, "CTS_Stor_mod_Wang.csv"), delimiter=",",skiprows=1)
    CTS_asc = np.loadtxt(os.path.join(data_path, "CTS_Stor_mod_Asch.csv"), delimiter=",",skiprows=1)

    #CTS_obs = [CTS_csv[:, 0], CTS_csv[:, 1]]

    X, _ = np.meshgrid(x, np.arange(Nz))
    

    plt.figure(figsize=(9,5))

    plt.plot(x, usurf, color="k", linewidth=1.5)
    plt.plot(x, topg, color="k", linewidth=1.5)
    plt.plot(CTS_obs[:,0], CTS_obs[:,1], color="k", linewidth=1.5, linestyle="--",label="Observed CTS")
    plt.plot(CTS_igm[:,0], CTS_igm[:,1], color='#0389bb', linewidth=1.5, linestyle="--",  label="IGM")
    plt.plot(CTS_wng[:,0], CTS_wng[:,1], color='#32a852', linewidth=1.5, linestyle="--",  label="Wang et al. ")
    plt.plot(CTS_asc[:,0], CTS_asc[:,1], color='#b01c3f', linewidth=1.5, linestyle="--", label="Aschwanden et al.")


    plt.legend()
    plt.xlabel(" Distance from Bergschrund (m)", fontsize=16)
    plt.ylabel("Elevation (m)", fontsize=16)
    plt.title("Comparaison of CTS positions", fontsize=18)
    plt.grid(True)
    plt.savefig(f"Plots_finaux/CTS_somparisaon.png")

    plt.show()

    

    

x = state.x.numpy()
z = state.z.numpy()
 
ix = 35
iy = 0

topg = state.topg  
thk = state.thk
usurf = state.usurf 

CTS= [x_CTS, CTS_1d]

#plot_profile_vitesse(U=U,z=z,topg=topg,usurf=usurf,ix=ix,iy=iy,component="u")
#plot_CTS(state)
#plot_vel(state)
_plot_results_E(times, T, drainage, U, W, E, E_pmp, strain_heat, omega, x, z)
_plot_results_T(times,topg, usurf, T, drainage, U, W, E, strain_heat, omega, x, z, CTS)
_plot_results_omega(times, T, drainage, U, W, E, E_pmp, strain_heat, omega, x, z, CTS)
