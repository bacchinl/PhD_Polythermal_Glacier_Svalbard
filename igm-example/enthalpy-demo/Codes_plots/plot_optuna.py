import optuna
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import numpy as np

study_name="example_ragna_mariebreen"

###### DATA ####
study = optuna.load_study(
    study_name=study_name,
    storage="sqlite:///../example_optuna_ragna_CTS.db",
)


df = study.trials_dataframe()
df = df[df["state"] == "COMPLETE"]
df = df[df["params_processes.time.start"].notna()]
df = df[df["value"] < 1e5]


cols = [
    "value",
    "params_processes.clim_load_climate.T_over_ela",
    "params_processes.clim_load_climate.ela",
    "params_processes.time.start",
    ]

df = df.dropna(subset=cols)

df["spinup_duration"] = 2024 - df["params_processes.time.start"]

print(df.columns)
print(df.head())
print("Number of simulation", len(df["number"]))

########### FUNCTIONS


def plot_3d_parameters(df):
    """
    Scatter 3D des essais Optuna.

    Axes :
        X : refreezing_water
        Y : ELA
        Z : basal_heat_flux

    Couleur :
        RMSE CTS (vert = faible, rouge = forte)
    """

    x = df["params_processes.clim_load_climate.T_over_ela"]
    y = df["params_processes.clim_load_climate.ela"]
    z = df["spinup_duration"]

    # suivant la version d'Optuna
    if "value" in df.columns:
        rmse = df["value"]
    else:
        rmse = df["values_0"]

    bounds = np.arange(10, 50, 5)
    norm = colors.BoundaryNorm(bounds, ncolors=256)


    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x,
        y,
        z,
        c=rmse,
        cmap="RdYlGn_r",   # vert = faible RMSE
        norm=norm,
        s=35,
        edgecolor="k",
        linewidth=0.3,
        alpha=0.9,
    )

    

    best = df.nsmallest(10, "value")

    ax.scatter(
        best["params_processes.clim_load_climate.T_over_ela"],
        best["params_processes.clim_load_climate.ela"],
        best["spinup_duration"],
        s=35,
        facecolors="darkgreen",
        edgecolors="black",
        linewidths=2,
        alpha=0.9,
    )

    cbar = plt.colorbar(sc, ax=ax, pad=0.12)
    cbar.set_label("CTS RMSE (m)", fontsize=12)

    ax.set_xlabel("Temperature over ELA")
    ax.set_ylabel("ELA (m)")
    ax.set_zlabel("Simulation duration (y)")

    ax.set_title("Optuna parameter search")

    plt.tight_layout()
    plt.show()

def plot_parameter_pairs(df):
    """
    Plot des trois couples de paramètres Optuna.

    Couleur = CTS RMSE.
    """

    # ---------- paramètres ----------
    refreezing = df["params_processes.clim_load_climate.T_over_ela"]
    ela = df["params_processes.clim_load_climate.ela"]
    time_start = df["spinup_duration"]

    rmse = df["value"]


    # ---------- palette discrète ----------
    bounds =  bounds = np.arange(10, 50, 5)

    norm = colors.BoundaryNorm(
        bounds,
        ncolors=256)

    cmap = plt.get_cmap("RdYlGn_r")

    # ---------- figure ----------
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18,5),
        constrained_layout=True,
    )

    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    # on enlève le 4ème subplot
    #axes[1,1].axis("off")
    best = df.nsmallest(5, "value")
    

    s = 45

    sc = ax1.scatter(
        refreezing,
        ela,
        c=rmse,
        cmap=cmap,
        norm=norm,
        s=s,
        edgecolor="k",
        linewidth=0.3,
    )

    ax2.scatter(
        refreezing,
        time_start,
        c=rmse,
        cmap=cmap,
        norm=norm,
        s=s,
        edgecolor="k",
        linewidth=0.3,
    )

    ax3.scatter(
        ela,
        time_start,
        c=rmse,
        cmap=cmap,
        norm=norm,
        s=s,
        edgecolor="k",
        linewidth=0.3,
    )

    ax1.scatter(
        best["params_processes.clim_load_climate.T_over_ela"],
        best["params_processes.clim_load_climate.ela"],
        s=s,
        facecolors="darkgreen",
        edgecolors="black",
        linewidths=2,
    )

    ax2.scatter(
        best["params_processes.clim_load_climate.T_over_ela"],
        best["spinup_duration"],
        s=s,
        facecolors="darkgreen",
        edgecolors="black",
        linewidths=2,
    )

    ax3.scatter(
        best["params_processes.clim_load_climate.ela"],
        best["spinup_duration"],
        s=s,
        facecolors="darkgreen",
        edgecolors="black",
        linewidths=2,
    )


    

    # ---------- labels ----------
    ax1.set_xlabel("Refreezing water fraction")
    ax1.set_ylabel("ELA (m)")

    ax2.set_xlabel("Refreezing water fraction")
    ax2.set_ylabel("Simulation duration")

    ax3.set_xlabel("ELA (m)")
    ax3.set_ylabel("Simulation duration")

    ax1.set_title("Refreezing vs ELA")
    ax2.set_title("Refreezing vs Time")
    ax3.set_title("ELA vs Time")

    # ---------- colorbar commune ----------
    cbar = fig.colorbar(
        sc,
        ax=axes,
        shrink=0.85,
        ticks=bounds[:-1],
        extend="max",
    )

    cbar.set_label("CTS RMSE (m)")
    plt.savefig(f"../optuna/{study_name}_params_compa.png")
    plt.show()


def plot_rmse_vs_parameters(df):

    # ---------- paramètres ----------
    refreezing = df["params_processes.clim_load_climate.T_over_ela"]
    ela = df["params_processes.clim_load_climate.ela"]
    duration = df["spinup_duration"]

    if "value" in df.columns:
        rmse = df["value"]
    else:
        rmse = df["values_0"]

    # ---------- palette ----------

    bounds =  bounds = np.arange(10, 50, 5)

    norm = colors.BoundaryNorm(
        bounds,
        ncolors=256)

    norm = colors.BoundaryNorm(bounds, 256, extend="max")
    cmap = plt.get_cmap("RdYlGn_r")

    fig, axes = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)

    params = [
        (refreezing, "Temperature over ELA"),
        (ela, "ELA (m)"),
        (duration, "Spin-up duration (yr)")
    ]
    best = df.nsmallest(10, "value")
    for ax, (x, xlabel) in zip(axes, params):

        sc = ax.scatter(
            x,
            rmse,
            c=rmse,
            cmap=cmap,
            norm=norm,
            s=45,
            edgecolor="k",
            linewidth=0.3,
        )


        # meilleure simulation
        ind = np.argmin(rmse)
        ax.axvline(x.iloc[ind], color="k", ls="--", lw=1)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("CTS RMSE (m)")
        ax.grid(alpha=0.3)

    cbar = fig.colorbar(
        sc,
        ax=axes,
        shrink=0.8,
        ticks=bounds[:-1],
        extend="max",
    )
    cbar.set_label("CTS RMSE (m)")
    plt.savefig(f"../optuna/{study_name}_rmse_vs_params.png")
    plt.show()


######## Call function
best = df.nsmallest(10, "value")
print("="*20, "10 BESTS SIMUS", "="*20)
print("VALUES    : ",best["value"])
print("ELA       : ",best["params_processes.clim_load_climate.ela"])
print("REFR WATER: ",best["params_processes.clim_load_climate.T_over_ela"])
print("TIME      : ",best["spinup_duration"])

plot_3d_parameters(df)
plot_parameter_pairs(df)
plot_rmse_vs_parameters(df)
