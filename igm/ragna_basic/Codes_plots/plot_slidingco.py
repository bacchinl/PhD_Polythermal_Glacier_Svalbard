import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator, splprep, splev

# ----------------------------
# Paths and output directory
# ----------------------------
date_simu = "2026-02-12/09-24-16/"


simu_path = os.path.join( "../outputs", date_simu)
out_dir = os.path.join("../outputs", date_simu, "Plots/sliding_co")
os.makedirs(out_dir, exist_ok=True)

which_flowline = "werenskiold_south"
flowline_file = os.path.join("../data", f"flowline_{which_flowline}_2022.csv")


point_fin = 5.7e3 # en m
smooth = True
# ----------------------------
# Load flowline from CSV
# ----------------------------
flowline_df = pd.read_csv(flowline_file)
x_flow = flowline_df['x'].values
y_flow = flowline_df['y'].values


# Smooth the line
if smooth :
    tck, _ = splprep(np.vstack([x_flow, y_flow]), s=0, k=min(3, len(x_flow)-1))
    n_samples = 200
    u_new = np.linspace(0, 1, n_samples)
    x_flow, y_flow = splev(u_new, tck)



# Compute cumulative distance along flowline
dist_flow = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x_flow)**2 + np.diff(y_flow)**2))])

# Find closest index to the target distance
idx_point_fin = np.argmin(np.abs(dist_flow - point_fin))
x_point_fin = x_flow[idx_point_fin]
y_point_fin = y_flow[idx_point_fin]
print(f"Point at {point_fin/1000:.2f} km from start:")
print(f"  x = {x_point_fin:.2f}, y = {y_point_fin:.2f}")

# ----------------------------
# Load model output
# ----------------------------
ds = xr.open_dataset(os.path.join(simu_path, "output.nc"))
sliding_co = ds['slidingco']
thk = ds['thk']
x = ds['x'].values
y = ds['y'].values
time = ds['time']

# Convert time to years (robust)
years = np.array([int(t) for t in time.values])

ntime = len(years)
print(f"{ntime} time steps loaded, example years: {years[:5]} ... {years[-5:]}")

# ----------------------------
# Ensure y is monotonically increasing
# ----------------------------
reverse_y = False
if y[0] > y[-1]:
    y = y[::-1]
    reverse_y = True

# ----------------------------
# Interpolate velocity along flowline
# ----------------------------
slide_along_flow = np.full((ntime, len(x_flow)), np.nan)

for it in range(ntime):
    slide_t = sliding_co.isel(time=it).values
    if reverse_y:
        slide_t = slide_t[::-1, :]

    interp = RegularGridInterpolator((y, x), slide_t,
                                     bounds_error=False, fill_value=np.nan)
    pts = np.vstack([y_flow, x_flow]).T
    slide_along_flow[it, :] = interp(pts)

print("Sliding coefficient interpolation along the flowline completed. Shape:", slide_along_flow.shape)

# ----------------------------
# Plot velocity profiles along flowline (every other year)
# ----------------------------
years_to_plot = years[::2]
cmap = plt.colormaps.get('turbo_r')
norm = mcolors.Normalize(vmin=years_to_plot.min(), vmax=years_to_plot.max())

plt.figure(figsize=(10, 6))
for yr in years_to_plot:
    idxs = np.where(years == yr)[0]
    if len(idxs) == 0:
        continue
    idx = idxs[0]
    # Limiter la distance jusqu’au point_fin
    dist_mask = dist_flow <= point_fin

    plt.plot(dist_flow[dist_mask] / 1000.0,
             slide_along_flow[idx, dist_mask],
             color=cmap(norm(yr)),
             label=str(yr),
             linewidth=2)

    #plt.plot(np.linspace(0, 1, len(x_flow)) * np.sum(np.sqrt(np.diff(x_flow)**2 + np.diff(y_flow)**2)) / 1000,
     #        speed_along_flow[idx, :],
      #       color=cmap(norm(yr)),
       #      label=str(yr),
        #     linewidth=2)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=plt.gca(), label="Year")
plt.xlabel("Distance along flowline (km)")
plt.ylabel("sliding coefficient (km MPa$^{-3}$ a$^{-1}$)")
plt.title(f"Sliding coefficient along the flowline {which_flowline}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"sliding_co_along_{which_flowline}_flowline.png"), dpi=200)
plt.show()

# ----------------------------
# Plot velocity field + normalized quiver + flowline
# ----------------------------
i_last = -1  # last time step
slide_last = sliding_co.isel(time=i_last).values
thk_last = thk.isel(time=i_last).values

if reverse_y:
    slide_last = slide_last[::-1, :]
    y_plot = y
else:
    y_plot = y

slide_valid = slide_last[thk_last > 0]

val_unique, counts = np.unique(slide_last, return_counts=True)
valeur_la_plus_frequente = val_unique[np.argmax(counts)]

moyenne = np.mean(slide_valid)
ecart_type = np.std(slide_valid)
min_slide = np.min(slide_valid)
max_slide = np.max(slide_valid)


print("Valeur la plus fréquente :", valeur_la_plus_frequente)
print("Moyenne :", moyenne)
print("Écart-type :", ecart_type)
print("Le coefficient de sliding prend comme valeur minimale ", min_slide," et comme valeur maximale ", max_slide)

alpha = np.where(thk_last>0, 1.0, 0)
plt.figure(figsize=(10, 8))
plt.pcolormesh(x, y_plot, slide_last, shading='auto', cmap='coolwarm', alpha=alpha, vmin=0, vmax=0.4)
plt.colorbar(label="sliding coefficient (km MPa$^{-3}$ a$^{-1}$)")


plt.title(f"Sliding coefficient field – Year {years[i_last]}")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"slindingco_map_{years[i_last]}.png"), dpi=200)
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(slide_valid, bins=40)
plt.xlabel("Valeurs de slide_valid")
plt.ylabel("Fréquence")
plt.title("Répartition des valeurs de slide_valid")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"slindingco_repartition_{years[i_last]}.png"), dpi=200)
plt.show()
