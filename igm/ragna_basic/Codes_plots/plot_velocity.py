import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin, xy
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import generic_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp


simu_path = "../outputs/2026-03-05/12-17-12/"
out_dir = f"{simu_path}/Plots/"
os.makedirs(out_dir, exist_ok=True)

PLOT_EPAISSEURS = True


year = 2021
SHOW = False


# Points de départ et d'arrivée
x0, y0 = 50000, 20000   # exemple (m)
x1, y1 = 80000, 45000   # exemple (m)

# --- Charger les données ---
# Charger le modèle (NetCDF)
ds = xr.open_dataset(f"{simu_path}/output.nc")
print(ds)
ds_in = xr.open_dataset("../data/input.nc")
icemask = ds_in["icemask"]



# Sélectionner la variable d'épaisseur (remplace 'epaisseur' par le vrai nom si différent)
x = ds['x'].values
y = ds['y'].values
print(x[0], x[-1])
print(y[0], y[-1])

u_var = ds['uvelsurf']   # (time, y, x)
v_var = ds['vvelsurf'] 
time = ds['time']
print("Variables importées")

for i in range(0, len(time)):
    year_i = int(time[i].values)
    if year_i == year :
        print(f"Year {year} found")
        u=u_var.isel(time=i)
        v=v_var.isel(time=i)
        #x=x_var.isel(time=i)
        #y=y_var.isel(time=i)

# Si le tableau est (y,x), les tailles doivent correspondre :
print("Shape u:", u.shape)
print("len(x):", len(x), "len(y):", len(y))


# === CALCUL DE LA VITESSE ===
speed = np.sqrt(u**2 + v**2)


# Si nécessaire, vérifier orientation
# (souvent les NetCDF ont y décroissant)
if y[0] > y[-1]:
    y = y[::-1]
    u = u[:, ::-1]
    v = v[:, ::-1]


# === VISUALISATION ===
plt.figure(figsize=(8,6))
plt.pcolormesh(x, y, speed, shading="auto", vmin=0, vmax=1)
plt.colorbar(label="Vitesse [m/an]")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Amplitude du champ de vitesse du glacier")
plt.axis("equal")
plt.show()



# === INTERPOLATION DU CHAMP DE VITESSE ===
U_interp = RegularGridInterpolator((x, y), u.T, bounds_error=False, fill_value=np.nan)
V_interp = RegularGridInterpolator((x, y), v.T, bounds_error=False, fill_value=np.nan)


def flowline_eq(t, XY):
    """Équations différentielles de la ligne de courant."""
    X, Y = XY
    u_val = U_interp((X, Y))
    v_val = V_interp((X, Y))
    return [u_val, v_val]

# === INTÉGRATION ===
# Durée d'intégration arbitraire ; adaptée à la taille du glacier
t_span = (0, 2e5)  # secondes ou unités arbitraires
# Stop si on sort du domaine
def event_out_of_bounds(t, XY):
    X, Y = XY
    if (X < x.min() or X > x.max() or Y < y.min() or Y > y.max()):
        return 0
    return 1
event_out_of_bounds.terminal = True



# === MESHGRID ===
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(10,8))
step = 10  # réduire le nombre de flèches pour la lisibilité
plt.quiver(X[::step, ::step], Y[::step, ::step], u[::step, ::step].T, v[::step, ::step].T, scale=10)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Champ de vitesse horizontal du glacier")
plt.axis("equal")
plt.show()



# Intégration directe (du point 1 → 2)
sol_forward = solve_ivp(flowline_eq, t_span, [x0, y0], max_step=200, events=event_out_of_bounds)

# === VISUALISATION ===
plt.figure(figsize=(8,6))
plt.quiver(x[::10], y[::10], u[::10, ::10].T, v[::10, ::10].T, alpha=0.4)
plt.plot(sol_forward.y[0], sol_forward.y[1], 'r-', lw=2, label="Flowline")
plt.scatter([x0, x1], [y0, y1], c=['g','k'], label='Start/End')
plt.legend()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title("Flowline entre deux points")
if SHOW :
    plt.show()
