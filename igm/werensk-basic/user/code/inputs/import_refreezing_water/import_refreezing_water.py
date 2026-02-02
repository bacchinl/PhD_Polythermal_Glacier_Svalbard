import xarray as xr
import numpy as np
import tensorflow as tf


def run(cfg, state):

    filepath = state.original_cwd.joinpath(
        cfg.core.folder_data,
        cfg.inputs.import_refreezing_water.filename
    )

    with xr.open_dataset(filepath) as f:
        ds = f.load()

    # -------------------------
    # Vérifications de base
    # -------------------------
    required_vars = ["internal_accumulation","refreeze"]
    for v in required_vars:
        if v not in ds:
            raise ValueError(f"Variable '{v}' not found in dataset")

    if "time" not in ds.dims:
        raise ValueError("Dataset must contain a 'time' dimension")

    # -------------------------
    # Choix de la variable
    # -------------------------
    varname = cfg.inputs.import_refreezing_water.variable  # ex: "refreeze" ou "internal_accumulation"
    data = ds[varname]

    # Option : appliquer le masque glacier
    if "gmask" in ds:
        data = data.where(ds.gmask > 0)

    # Remplacer FillValue
    data = data.where(np.isfinite(data), 0.0)

    # -------------------------
    # Conversion vers TensorFlow
    # -------------------------
    # shape = (time, y, x)
    state.water_refreeze = tf.Variable(
        data.values.astype("float32"),
        name="water_refreeze"
    )

    # Axe temps (jours depuis 1991-01-01)
    state.time_refreeze = tf.constant(
        ds.time.values.astype("float32"),
        name="time_refreeze"
    )

    # Coordonnées spatiales utiles
    if "lat" in ds:
        state.lat = tf.constant(ds.lat.values.astype("float32"))
    if "lon" in ds:
        state.lon = tf.constant(ds.lon.values.astype("float32"))

    # Meta pour output
    state.ds_meta_only = xr.Dataset(attrs=ds.attrs)

    if hasattr(state, "logger"):
        state.logger.info(
            f"Imported {varname} with shape {state.water_refreeze.shape}"
        ) 
