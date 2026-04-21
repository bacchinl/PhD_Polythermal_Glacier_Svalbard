import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
import pandas as pd


# --- Parameters ---
simu_path = "../outputs/2025-10-30/12-57-09/"
out_dir = f"{simu_path}/Plots/"
os.makedirs(out_dir, exist_ok=True)


which_flowline = "werenskiold_north"

if which_flowline == "werenskiold_south" :
# Starting and ending points (in meters)
    x0, y0 = 512200, 8554420
    x1, y1 = 506265, 8556140

if which_flowline == "werenskiold_north" :
# Starting and ending points (in meters)
    x0, y0 = 510500, 8558350
    x1, y1 = 506265, 8556140



# Year of interest
target_year = 2022

# --- Load the dataset ---
ds = xr.open_dataset(f"{simu_path}/output.nc")

x = ds["x"].values
y = ds["y"].values
time = ds["time"].values
u_var = ds["uvelsurf"]  # (time, y, x)
v_var = ds["vvelsurf"]

# --- Find the index for the desired year ---

years = np.array([int(t) for t in time])

idx_time = np.where(years == target_year)[0][0]
print(f"Using year {target_year}")

# Extract fields
u = u_var.isel(time=idx_time).values
v = v_var.isel(time=idx_time).values

# Fix y orientation if needed
reverse_y = False
if y[0] > y[-1]:
    y = y[::-1]
    u = u[::-1, :]
    v = v[::-1, :]
    reverse_y = True

speed = np.sqrt(u**2 + v**2)

# --- Prepare interpolators ---
U_interp = RegularGridInterpolator((y, x), u, bounds_error=False, fill_value=np.nan)
V_interp = RegularGridInterpolator((y, x), v, bounds_error=False, fill_value=np.nan)

# === Define the differential equation for streamline ===
def flow_eq(s, XY):
    X, Y = XY
    u_val = U_interp((Y, X))
    v_val = V_interp((Y, X))
    # Normalize to follow direction only (unit vector)
    norm = np.sqrt(u_val**2 + v_val**2)
    if np.isnan(norm) or norm == 0:
        return [0, 0]
    return [u_val / norm, v_val / norm]

# === Integration parameters ===
# We'll integrate along the flow direction (forward)
s_max = 20000  # integration distance (in m) — adjust as needed
ds_step = 10   # step size (m)

# Integration with solve_ivp
sol = solve_ivp(
    flow_eq,
    t_span=(0, s_max),
    y0=[x0, y0],
    method="RK45",
    max_step=ds_step,
    dense_output=True
)

x_flow = sol.y[0]
y_flow = sol.y[1]

# Optionally: stop when leaving domain
mask = (x_flow >= x.min()) & (x_flow <= x.max()) & (y_flow >= y.min()) & (y_flow <= y.max())
x_flow = x_flow[mask]
y_flow = y_flow[mask]

print(f"Computed flowline with {len(x_flow)} points.")
u_dir = np.copy(u)
v_dir = np.copy(v)
norm = np.sqrt(u_dir**2 + v_dir**2)
norm[norm == 0] = np.nan  # éviter les divisions par 0
u_dir /= norm
v_dir /= norm

# === Plot ===
plt.figure(figsize=(10, 8))

# Background speed field
plt.pcolormesh(x, y, speed, shading="auto", cmap="viridis")
plt.colorbar(label="Vitesse (m/an)")

# Quiver field (downsampled)
X, Y = np.meshgrid(x, y)
step = 10
#factor = 50  # amplify small arrows visually
plt.quiver(X[::step, ::step], Y[::step, ::step],
           u_dir[::step, ::step],
           v_dir[::step, ::step],
           scale=40, color="k", width=0.002, alpha=0.6)

# Flowline
plt.plot(x_flow, y_flow, "r-", linewidth=2.5, label="Flowline calculée")
plt.scatter([x0], [y0], c="cyan", s=60, label="Départ")
plt.scatter([x1], [y1], c="orange", s=60, label="Référence (x1, y1)")

plt.title(f"Flowline calculée suivant le flux – Année {target_year}")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"flowline_{which_flowline}_{target_year}.png"), dpi=200)
plt.show()


flowline_df = pd.DataFrame({'x': x_flow, 'y': y_flow})
flowline_file = os.path.join(out_dir, f"flowline_{which_flowline}_{target_year}.csv")
flowline_df.to_csv(flowline_file, index=False)
print(f"Flowline saved to {flowline_file}")
