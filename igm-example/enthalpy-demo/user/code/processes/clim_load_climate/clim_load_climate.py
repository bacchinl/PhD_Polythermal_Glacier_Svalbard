#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, sys, shutil
import time
import tensorflow as tf
from scipy.interpolate import RectBivariateSpline, interp1d
from igm.utils.math.interp1d_tf import interp1d_tf
import igm
import xarray as xr


def initialize(cfg, state):

    # Check if an array of time-dependent temp and precip offset are provided by user (and initialize the variable to store it if yes)
    if cfg.processes.clim_load_climate.climate_change_array == []:
        state.climatepar = None
    else:
        state.climatepar = np.array(cfg.processes.clim_load_climate.climate_change_array[1:]).astype(np.float32)
 
    # Create the Usurf_initial from which to compute offsets in future Usurf during update
    state.usurf_initial = state.usurf

    produce_climate_data(cfg, state)


    state.air_temp = tf.Variable(state.air_temp_ref, dtype="float32")
    state.air_temp_sd = tf.Variable(state.air_temp_sd_ref, dtype="float32")
    
    # create the time loggers 
    state.tcomp_clim_load_climate = []
    state.tlast_clim_update = tf.Variable(-1.0e5000)
    

def update(cfg, state):
    """Update air temperature and precipitation based on modelled ice surface elevation changes."""

    # update climate fields each X years
    if (state.t - state.tlast_clim_update) >= cfg.processes.clim_load_climate.update_freq:
        if hasattr(state, "logger"):
            state.logger.info(
                "Construct climate at time : " + str(state.t.numpy())
            )

        state.tcomp_clim_load_climate.append(time.time())

        # Explicitly reset state temp and precip variables to their initial values before applying changes (otherwise we can get a runaway effect...)
        state.air_temp.assign(state.air_temp_ref)  # Reset air temperature to initial reference temperature field

        ### If set by user, apply the time-dependent climate changes
        if cfg.processes.clim_load_climate.time_dependent_climate and state.climatepar is not None:
            # Interpolate temperature and precipitation offsets at current time
            temp_offset = interp1d_tf(state.climatepar[:, 0], state.climatepar[:, 1], state.t)
            temp_offset = tf.broadcast_to(temp_offset, tf.shape(state.air_temp))
            
            # Apply temperature offsets (in °C) for current model time
            state.air_temp.assign_add(temp_offset)


        ### Set T=cte degrees over the ela
        ela = cfg.processes.clim_load_climate.ela
        T_over_ela = cfg.processes.clim_load_climate.T_over_ela

        mask_above_ela = tf.cast(
            state.usurf > ela,
            state.air_temp.dtype
        )
        
        new_air_temp =new_air_temp = (
            state.air_temp * (1.0 - mask_above_ela)
            + T_over_ela * mask_above_ela
        )

        state.air_temp.assign(new_air_temp)




        # update the time loggers
        state.tlast_clim_update.assign(state.t)
        state.tcomp_clim_load_climate[-1] -= time.time()
        state.tcomp_clim_load_climate[-1] *= -1

        # Print climate data to make sure values make sense (optionnal)
        #tf.print("air_temp (°C): min =", tf.reduce_min(state.air_temp), "max =", tf.reduce_max(state.air_temp))
        #tf.print("air_temp_sd (°C): min =", tf.reduce_min(state.air_temp_sd), "max =", tf.reduce_max(state.air_temp_sd))
        
        #tf.print("Shape of delta_usurf:", tf.shape(delta_usurf))
        #tf.print("Shape of air_temp:", tf.shape(state.air_temp))

        #tf.print("delta_usurf: min =", tf.reduce_min(delta_usurf), "max =", tf.reduce_max(delta_usurf))
        
def finalize(cfg, state):
    pass



def produce_climate_data(cfg, state):

    # Load the mean annual air temperature field from state
    air_temp = state.air_temp

    # Load the mean summer air temperature field from state
    air_temp_summer = state.air_temp_summer


    # Create the space-invariant air_temp_sd variable
    air_temp_sd = tf.fill(tf.shape(state.thk), cfg.processes.clim_load_climate.air_temperature_stdev)


    # Expand dimensions to add time (e.g., 12 months with data every months)
    # At the moment the climate data is constant accross the 12 months
    time_steps = 12  # Set this based on your needs (e.g., monthly data)
    air_temp_expand = tf.expand_dims(air_temp, axis=0)  # Add time dimension
    air_temp_expand = tf.repeat(air_temp_expand, time_steps, axis=0)  # Repeat for all time steps

    air_temp_sd = tf.expand_dims(air_temp_sd, axis=0)
    air_temp_sd = tf.repeat(air_temp_sd, time_steps, axis=0)


    # Apply seasonal variation with cosine yearly cycle if enabled (sinusoidal fluctuation around the mean temp using a cosine function.)
    if cfg.processes.clim_load_climate.cosine_yearly_cycle_temp:
        # Compute per-pixel amplitude based on summer temp
        amplitude = air_temp_summer - air_temp  # shape: [lat, lon]
        amplitude = tf.expand_dims(amplitude, axis=0)  # shape [1, lat, lon]

        # Time steps
        months = np.arange(12)
        if cfg.processes.clim_load_climate.southern_hemisphere_climate:
            seasonal_cycle = np.cos(2 * np.pi * (months) / 12)  # Max in January
        else:
            seasonal_cycle = np.cos(2 * np.pi * (months - 6) / 12)  # Max in July

        seasonal_cycle = tf.convert_to_tensor(seasonal_cycle, dtype=tf.float32)  # [12]
        seasonal_cycle = tf.reshape(seasonal_cycle, (12, 1, 1))  # [12, 1, 1] for broadcasting

        # Add seasonal cycle
        air_temp_expand = air_temp_expand + amplitude * seasonal_cycle


    #######################   Save the climate as NetCDF (optional) #############
    if cfg.processes.clim_load_climate.export_climate_ref:
        ds = xr.Dataset(
            {
                "air_temp": (["time", "y", "x"], air_temp_expand.numpy()),
                "air_temp_sd": (["time", "y", "x"], air_temp_sd.numpy()),
            },
            coords={
                "time": np.arange(12),  # 12 months
                "x": state.x.numpy() if hasattr(state, "x") else np.arange(air_temp.shape[1]),
                "y": state.y.numpy() if hasattr(state, "y") else np.arange(air_temp.shape[0]),
            },
        )
        # Save to NetCDF file
        ds.to_netcdf("climate_ref.nc")
    ################################################################################


    # We shift the climate time dimension to represent the hydrological year (start of the year becomes start of the accumulation season)
    # Start of accumulation season is set as 1st November in Northern Hemisphere, and 1st May in Southern hemisphere.
    shift = 2 / 12 if not cfg.processes.clim_load_climate.southern_hemisphere_climate else 8 / 12
    # Apply shift using np.roll, this should work for any number of time steps in a year (monthly, weekly, daily data)
    shift_steps = int(time_steps * (1 - shift))
    air_temp_expand = tf.roll(air_temp_expand, shift=shift_steps, axis=0)
    air_temp_sd = tf.roll(air_temp_sd, shift=shift_steps, axis=0)



    # Produce the final fields for the update function
    state.air_temp_ref = tf.constant(air_temp_expand, dtype="float32")
    state.air_temp_sd_ref = tf.constant(air_temp_sd, dtype="float32")
