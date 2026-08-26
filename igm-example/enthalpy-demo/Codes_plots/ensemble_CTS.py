import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import optuna


##### PARAMS
nb_best_simu =5
date_simu = "2026-08-26/09-11-31"


data_dir = "../obs_cts/thickness_cts_csv"

radar_line = "ragna_mariebreen-20240412-DAT_0404_A1_1"
#radar_line = "ragna_mariebreen-20240412-DAT_0405_A1_1"
#radar_line = "ragna_mariebreen-20230305-DAT_0067_A1_1"
#radar_line = "ragna_mariebreen-20230305-DAT_0068_A1_3"



gpr_csv_path = os.path.join(
    data_dir,
    f"thickness_cts_points_{radar_line}.csv"
)


out_dir = os.path.join("../multirun/",date_simu, f"Plots/Ice_type_profile/{radar_line}")
os.makedirs(out_dir, exist_ok=True)


#### FUNCTION


def compute_cts(E3d, Epmp3d, topg, thk, zeta_mid):

    nz, npts = E3d.shape

    cts = np.full(npts, np.nan)

    for p in range(npts):

        delta = E3d[:, p] - Epmp3d[:, p]

        ind = np.where(np.diff(np.sign(delta)) != 0)[0]

        if len(ind) == 0:
            continue

        k = ind[0]

        z = topg[p] + zeta_mid * thk[p]

        d1 = delta[k]
        d2 = delta[k + 1]

        z1 = z[k]
        z2 = z[k + 1]

        if np.abs(d2 - d1) < 1e-12:
            cts[p] = z1
        else:
            cts[p] = z1 - d1 * (z2 - z1) / (d2 - d1)

    return cts




### Run


study = optuna.load_study(
    study_name="example_ragna_mariebreen",
    storage="sqlite:///../example_optuna_ragna_CTS.db",
)

df = study.trials_dataframe()

df = df[df["state"] == "COMPLETE"]
df = df[df["value"] < 1e5]

best = df.nsmallest(nb_best_simu, "value")


##### LOAD OBS
gpr = pd.read_csv(gpr_csv_path)

x_flow = gpr["easting"].values
y_flow = gpr["northing"].values
cts_obs = gpr["temperate_elevation"].values
cts_obs_low = gpr["temperate_lower"].values
cts_obs_up = gpr["temperate_upper"].values
cts_err = gpr["temperate_gpr_uncertainty"].values

dist = gpr["distance"].values / 1000



#### LOOP on best simu

cts_all = []

for trial in best["number"]:

    path = os.path.join(
        "../multirun",
        date_simu,
        str(trial),
        "output.nc",
    )

    ds = xr.open_dataset(path)

    E = ds.E.isel(time=-1).values
    Epmp = ds.E_pmp.isel(time=-1).values

    usurf = ds.usurf.isel(time=-1).values
    topg = ds.topg.isel(time=-1).values
    thk = ds.thk.isel(time=-1).values

    x = ds.x.values
    y = ds.y.values
    z = ds.z.values

    nz = len(z)

    vert_spacing = 4

    zeta_edges = np.arange(nz + 1) / nz
    zeta_edges = (
        zeta_edges / vert_spacing
    ) * (1 + (vert_spacing - 1) * zeta_edges)

    zeta_mid = 0.5 * (
        zeta_edges[:-1] + zeta_edges[1:]
    )

    interpE = RegularGridInterpolator((z, y, x), E)
    interpEp = RegularGridInterpolator((z, y, x), Epmp)

    interpTopg = RegularGridInterpolator((y, x), topg)
    interpThk = RegularGridInterpolator((y, x), thk)
    interpSurf = RegularGridInterpolator((y, x), usurf)

    pts2d = np.c_[y_flow, x_flow]

    topg_f = interpTopg(pts2d)
    thk_f = interpThk(pts2d)
    surf_f = interpSurf(pts2d)

    Eprof = np.empty((nz, len(x_flow)))
    Epprof = np.empty_like(Eprof)

    for k in range(nz):

        pts3d = np.c_[
            np.full(len(x_flow), z[k]),
            y_flow,
            x_flow,
        ]

        Eprof[k] = interpE(pts3d)
        Epprof[k] = interpEp(pts3d)

    cts = compute_cts(
        Eprof,
        Epprof,
        topg_f,
        thk_f,
        zeta_mid,
    )

    cts_all.append(cts)


cts_all = np.asarray(cts_all)

cts_med = np.nanmedian(cts_all, axis=0)
cts_p10 = np.nanpercentile(cts_all, 10, axis=0)
cts_p90 = np.nanpercentile(cts_all, 90, axis=0)

##### Plot ##

plt.figure(figsize=(11,6))

# enveloppe
plt.fill_between(
    dist,
    cts_p10,
    cts_p90,
    color="tab:red",
    alpha=0.25,
    label="10 best simulations",
)

plt.fill_between(
    dist,
    topg_f+cts_obs_low,
    topg_f+cts_obs_up,
    color="tab:blue",
    alpha=0.25,
    label="Uncertainty in observation",
)


# toutes les simulations
for cts in cts_all:

    plt.plot(
        dist,
        cts,
        color="0.75",
        lw=1,
        alpha=0.4,
    )

# médiane
plt.plot(
    dist,
    cts_med,
    color="tab:red",
    lw=3,
    label="Median",
)

cts_obs = np.where(cts_obs < surf_f, cts_obs, np.nan)
cts_obs = np.where(cts_obs > topg_f, cts_obs, np.nan)
# observations
plt.plot(
    dist,
    cts_obs,
    "b--",
    lw=2,
    label="Observed CTS",
)

# surface
plt.plot(
    dist,
    surf_f,
    "k",
    lw=2,
)

# bedrock
plt.plot(
    dist,
    topg_f,
    "k",
)

plt.fill_between(
    dist,
    topg_f,
    np.min(topg_f)-30,
    color="saddlebrown",
)

plt.xlabel("Distance along profile (km)")
plt.ylabel("Elevation (m a.s.l.)")

plt.legend()

plt.tight_layout()

fname = os.path.join(out_dir, f"cts_elevation_{nb_best_simu}_optuna.png")
plt.savefig(fname, dpi=200)
plt.show()
