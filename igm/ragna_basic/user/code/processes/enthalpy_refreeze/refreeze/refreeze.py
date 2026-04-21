from omegaconf import DictConfig
from igm.common import State

def compute_refreezing_heat(cfg: DictConfig, state: State) -> None:

    cfg_thermal = cfg.processes.enthalpy_refreeze.thermal
    
    L_ice = cfg_thermal.L_ice

    E = state.E + L_ice*state.refreezing_water

    return E
