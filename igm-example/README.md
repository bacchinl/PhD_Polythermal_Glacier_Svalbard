#Enthalpy module demonstration - Dronbreen
**A specific example get familiar with the enthalpy module capacities.**


This example is divided into two parts:

- A first simple forward modelling run with the enthalpy module activated.
- A data assimilation multirun using Optuna to fit observed cold–temperate transition surfaces (CTS) from Mannerfelt et al. *submitted* (https://zenodo.org/records/17882300).

Both experiments are performed on Dronbreen, a polythermal glacier located in Svalbard with a large number of available GPR measurements. The input file has a 20 m spatial resolution, with ice thickness reconstructed using kriging.

## Step 1 - Forward modelling with the enthalpy module

The first experiment is a spin-up with a fixed geometry and a constant climate derived from CARRA.

In a real Scandinavian-type polythermal glacier, temperate ice can form through the refreezing of surface meltwater in the firn. As this process is not yet explicitly implemented in IGM, this example uses a simplified approach: a constant temperature is prescribed above the Equilibrium Line Altitude (ELA) to represent the heat released by meltwater refreezing.

The temperature above the ELA is set with: 

```yaml
processes.clim_load_climate.T_over_ela
```
and the ela with

```yaml
processes.clim_load_climate.ela
```

Run the experiment with `igm_run +experiment=params_spin_up_enthalpy`



## Step 2 - Finding the best parameters with Optuna

The second experiment uses *Optuna* to find the parameters that best reproduce the observed CTS from GPR measurements.

Optuna will optimises three parameters (Temperature over ELA, ELA and simulation duration) to find the lowest RMSE between the modelled CTS and the observed CTS. For each trial, Optuna proposes a new combination of these parameters, IGM runs the simulation, computes the modelled CTS, and returns the CTS RMSE.

`
igm_run -m \
    +experiment=params_spin_up_optuna \
    hydra/sweeper=igm_optuna \
    hydra.sweeper.optuna_config=optuna/example_optuna_Dronbreen_CTS.yaml`




## Vizualitation tools :

For optuna : `python plot_optuna.py` or `optuna-dashboard sqlite:///example_optuna_Dronbreen_CTS.db` and connect to the given link with your browser. 



