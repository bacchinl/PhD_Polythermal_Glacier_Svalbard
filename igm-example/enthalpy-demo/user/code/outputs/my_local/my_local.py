import time
import xarray as xr
import numpy as np
import tensorflow as tf
import os
from netCDF4 import Dataset

from igm.utils.math.getmag import getmag

def initialize(cfg, state):
    state.var_info_ncdf_ex = {
        "topg": ["Basal Topography", "m"],
        "usurf": ["Surface Topography", "m"],
        "thk": ["Ice Thickness", "m"],
        "icemask": ["Ice mask", "NO UNIT"],
        "smb": ["Surface Mass Balance", "m/y ice eq"],
     	"ubar": ["x depth-average velocity of ice", "m/y"], 
    	"vbar": ["y depth-average velocity of ice", "m/y"], 
 	    "velbar_mag": ["Depth-average velocity magnitude of ice", "m/y"], 
 	    "uvelsurf": ["x surface velocity of ice", "m/y"], 
 	    "vvelsurf": ["y surface velocity of ice", "m/y"], 
 	    "wvelsurf": ["z surface velocity of ice", "m/y"], 
 	    "velsurf_mag": ["Surface velocity magnitude of ice", "m/y"], 
 	    "uvelbase": ["x basal velocity of ice", "m/y"], 
 	    "vvelbase": ["y basal velocity of ice", "m/y"], 
 	    "wvelbase": ["z basal velocity of ice", "m/y"], 
 	    "velbase_mag": ["Basal velocity magnitude of ice", "m/y"], 
 	    "divflux": ["Divergence of the ice flux", "m/y"],
        "strflowctrl": ["arrhenius+1.0*slidingco", "MPa$^{-3}$ a$^{-1}$"],
        "dtopgdt": ["Erosion rate", "m/y"],
        "arrhenius": ["Arrhenius factor", "MPa$^{-3}$ a$^{-1}$"],
        "slidingco": ["Sliding Coefficient", "km MPa$^{-3}$ a$^{-1}$"],
        "meantemp": ["Mean annual surface temperatures", "°C"],
        "ela": ["Equilibrium Line Altitude", "m asl"], 
        "air_temp": ["Mean annual air temperatures", "°C"],
        "air_temp_summer":[ "Mean summer air temperatures", "°C"],
        "meanprec": ["Mean annual precipitation", "Kg m^(-2) y^(-1)"],
        "precipitation": ["Total precipitation per year (annual mean)", "mm/year"],
        "velsurfobs_mag": ["Obs. surf. speed of ice", "m/y"],
        "weight_particles": ["weight_particles", "no"],
        "T": ["Temperature of the ice", "K"],
        "T_pmp": ["Temperature of the pressure melting point", "K"],
        "E": ["Enthalpy", "J kg$^{-1}$"],
        "E_pmp": ["Enthalpy at the pressure melting point", "J kg$^{-1}$"],
        "omega": ["Fraction of water content", "-"],
        "strain_heat": ["Strain heating", "W m$^{-3}$"],
        "friction_heat": ["Friction heating", "W m$^{-2}$"],
        "refreezing_heat": ["Refreezing heating", "J kg$^{-1}$"]
    }

    state.var_info_ncdf_ts = {}
    state.var_info_ncdf_ts["vol"] = ["Ice volume", "km^3"]
    state.var_info_ncdf_ts["area"] = ["Glaciated area", "km^2"]

def run(cfg, state):

    if not state.saveresult:
        return

    # Prepare any derived quantities
    if "velbar_mag" in cfg.outputs.my_local.vars_to_save:
        state.velbar_mag = getmag(state.ubar, state.vbar)

    if "velsurf_mag" in cfg.outputs.my_local.vars_to_save:
        state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

    if "velbase_mag" in cfg.outputs.my_local.vars_to_save:
        state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

    if 'netcdf' in cfg.outputs.my_local.file_format_list:
        update_netcdf_ex(cfg,state)
    
    if 'tif' in cfg.outputs.my_local.file_format_list:
        write_tif(cfg,state)


#############################################

def write_tif(cfg,state):

    var_list = cfg.outputs.my_local.vars_to_save

    for var in var_list:
        if not hasattr(state, var):
                continue
        
        var_data = vars(state)[var].numpy()
        file_name = f"{var}-{str(getattr(state, 't', tf.constant(0)).numpy()).zfill(6)}.tif"

        data_array = xr.DataArray(
            var_data,
            dims=("y", "x"),
            coords={"y": state.y.numpy(), "x": state.x.numpy()}
        )

        if "crs" in cfg.outputs.my_local:
            data_array.rio.write_crs(cfg.outputs.my_local.crs, inplace=True)

        data_array.rio.to_raster(file_name)

#####################################

def update_netcdf_ex(cfg,state):
    
    
    file_path = cfg.outputs.my_local.output_file
    var_list = cfg.outputs.my_local.vars_to_save

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_time_value():
        """Return current model time as a scalar numpy value."""
        return getattr(
            state,
            "t",
            tf.constant(0)
        ).numpy()

    def to_numpy(value):
        """Convert TensorFlow tensor to numpy when necessary."""
        return value.numpy() if hasattr(value, "numpy") else value

    def create_data_vars():
        """
        Extract requested variables from state and convert them
        to numpy arrays with the appropriate dimensions.
        """

        data_vars = {}

        for var in var_list:

            if not hasattr(state, var):
                print(f"No {var} in State!")
                continue

            arr = vars(state)[var].numpy()

            # ----------------------------------------------------------
            # 2D variable: y, x
            # ----------------------------------------------------------
            if arr.ndim == 2:

                data = xr.DataArray(
                    arr,
                    dims=("y", "x"),
                )

            # ----------------------------------------------------------
            # Scalar variable
            # ----------------------------------------------------------
            elif arr.ndim == 0:

                data = xr.DataArray(
                    arr,
                    dims=(),
                )

            # ----------------------------------------------------------
            # 3D variable: z, y, x
            # ----------------------------------------------------------
            elif arr.ndim == 3:

                Nz = arr.shape[0]

                data = xr.DataArray(
                    arr,
                    dims=("z", "y", "x"),
                    coords={
                        "z": np.arange(Nz)
                    },
                )

            else:
                # Unsupported dimensionality
                continue

            # Add time dimension
            data = data.expand_dims(
                time=[get_time_value()]
            )

            # Add metadata if available
            if (
                hasattr(state, "var_info_ncdf_ex")
                and var in state.var_info_ncdf_ex
            ):
                data.attrs["long_name"], data.attrs["units"] = (
                    state.var_info_ncdf_ex[var]
                )

            data_vars[var] = data

        if cfg.outputs.my_local.add_thk_vol:

            thk = to_numpy(state.thk)
            dx = to_numpy(state.dx)

            vol = np.sum(thk) * (dx ** 2) / 1e9
            area = np.sum(thk > 1) * (dx ** 2) / 1e6

            
            data_vars["vol"] = xr.DataArray([vol],dims=("time"))

            data_vars["area"] = xr.DataArray([area],dims=("time"))

            # Metadata
            if hasattr(state, "var_info_ncdf_ts"):

                if "vol" in state.var_info_ncdf_ts:
                    data_vars["vol"].attrs["long_name"] = (
                        state.var_info_ncdf_ts["vol"][0]
                    )
                    data_vars["vol"].attrs["units"] = (
                        state.var_info_ncdf_ts["vol"][1]
                    )

                if "area" in state.var_info_ncdf_ts:
                    data_vars["area"].attrs["long_name"] = (
                        state.var_info_ncdf_ts["area"][0]
                    )
                    data_vars["area"].attrs["units"] = (
                        state.var_info_ncdf_ts["area"][1]
                    )

            # Fallback metadata
            data_vars["vol"].attrs.setdefault(
                "long_name",
                "Total ice volume",
            )
            data_vars["vol"].attrs.setdefault(
                "units",
                "10^9 km^3",
            )

            data_vars["area"].attrs.setdefault(
                "long_name",
                "Ice-covered area",
            )
            data_vars["area"].attrs.setdefault(
                "units",
                "10^6 km^2",
            )


        return data_vars

    # ------------------------------------------------------------------
    # Prepare variables
    # ------------------------------------------------------------------

    data_vars = create_data_vars()

    time_value = get_time_value()

    # ------------------------------------------------------------------
    # CASE 1: File does not exist -> create it
    # ------------------------------------------------------------------

    if not os.path.exists(file_path):

        if hasattr(state, "logger"):
            state.logger.info(
                f"Creating new NetCDF file: {file_path}"
            )

        # Build initial dataset
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "x": (
                    "x",
                    state.x.numpy()
                ),
                "y": (
                    "y",
                    state.y.numpy()
                ),
                "time": (
                    "time",
                    [time_value]
                ),
            },
            attrs={
                "pyproj_srs": getattr(
                    state,
                    "pyproj_srs",
                    ""
                )
            },
        )

        # --------------------------------------------------------------
        # Create a temporary file first.
        #
        # If the program crashes during creation, the existing/final
        # file is never touched.
        # --------------------------------------------------------------

        tmp_file = file_path + ".tmp"

        try:

            # Make sure an old temporary file doesn't interfere
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

            ds.to_netcdf(
                tmp_file,
                mode="w",
                unlimited_dims=["time"],
            )

            ds.close()

            # Atomic replacement
            os.replace(
                tmp_file,
                file_path
            )

        except Exception:

            # Close dataset if necessary
            try:
                ds.close()
            except Exception:
                pass

            # Remove incomplete temporary file
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass

            raise

    # ------------------------------------------------------------------
    # File already exists -> append time dim
    # ------------------------------------------------------------------

    else:

        if hasattr(state, "logger"):
            state.logger.info(
                f"Appending to NetCDF file at iteration "
                f"{getattr(state, 'it', '?')}"
            )

        # --------------------------------------------------------------
        # Open existing NetCDF in append mode
        # --------------------------------------------------------------

        with Dataset(file_path, mode="a") as nc:

            # ----------------------------------------------------------
            # Current number of timesteps
            # ----------------------------------------------------------

            time_index = len(nc.dimensions["time"])

            # ----------------------------------------------------------
            # Write time
            # ----------------------------------------------------------

            nc.variables["time"][time_index] = time_value

            # ----------------------------------------------------------
            # Write each variable
            # ----------------------------------------------------------

            for var, data_array in data_vars.items():

                if var not in nc.variables:
                    if hasattr(state, "logger"):
                        state.logger.warning(
                            f"Variable {var} not found in existing "
                            f"NetCDF file. Skipping."
                        )
                    continue

                arr = data_array.values

                if arr.ndim > 0 and arr.shape[0] == 1:
                    arr = arr[0]

                # Write new timestep
                nc.variables[var][time_index, ...] = arr

            # Make sure everything is flushed to disk
            nc.sync()


#########################################################
