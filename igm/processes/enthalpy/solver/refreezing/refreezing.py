#!/usr/bin/env python3

# Copyright (C) Lucie le morse



from omegaconf import DictConfig

from igm.common import State
from .utils import compute_heat_refreeze
import tensorflow as tf

def add_refreezing_heat_source(cfg: DictConfig, state: State) -> None:
    """
    Update enthalpy field for refreezing water in the firn over a time step.

    Add to the enthalpy field a term corresponding to the refreezing of percolating water in the firn.

    Updates state.E (J kg^-1).
    """
    depth = state.iceflow.discr_v.enthalpy.depth
    depth_firn = cfg.processes.enthalpy.surface.depth_firn

    firn_mask = depth <= depth_firn
    

    ela = 360 #cfg.processes.enthalpy.surface.ela #state.ELA 
    mask_above_ela = state.usurf > ela
    firn_mask = firn_mask & mask_above_ela[None, ...]

    print("Shape E : ",state.E.shape)
    print("Shape depth : ", depth.shape)
    print("Shape firn_mask : ", firn_mask.shape)
   
    print("firn cells:", tf.reduce_sum(tf.cast(firn_mask, tf.int32)))

    omega = 1
    frac_refreezing = 0.8
    L_ice = cfg.processes.enthalpy.thermal.L_ice

    heat =  omega*frac_refreezing*L_ice  #compute_heat_refreeze(frac_refreezing,omega,L_ice)
    state.refreezing_heat = heat * tf.cast(firn_mask, state.E.dtype)
    state.E = state.E + state.refreezing_heat

