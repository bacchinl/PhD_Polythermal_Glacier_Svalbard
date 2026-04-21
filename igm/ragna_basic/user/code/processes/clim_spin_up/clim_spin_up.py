#!/usr/bin/env python3

# Climate spin-up module for IGM
# Repeat one climate year over the whole simulation

import numpy as np
import os
import tensorflow as tf
import time


def params(parser):

    parser.add_argument(
        "--update_freq",
        type=float,
        default=1,
        help="Update the climate each X years",
    )

    parser.add_argument(
        "--time_resolution",
        type=float,
        default=365,
        help="Climate temporal resolution (365=daily, 12=monthly)",
    )

    parser.add_argument(
        "--name_file",
        type=str,
        default="climate.dat",
        help="Climate file name",
    )

    parser.add_argument(
        "--spin_up_year",
        type=int,
        default=None,
        help="Year to repeat for spin-up",
    )


def initialize(cfg, state):
    """
    Load climate data and select the spin-up year to repeat
    """

    # altitude of the weather station
    state.zws = 390

    # Load climate file
    temp_prec = np.loadtxt(
        state.original_cwd.joinpath(os.path.join("data", cfg.processes.clim_spin_up.name_file)),
        dtype=np.float32,
        skiprows=1,
    )

    # Get available years
    ymin = int(min(temp_prec[:, 0]))
    ymax = int(max(temp_prec[:, 0]))

    state.temp = np.zeros((365, ymax - ymin + 1), dtype=np.float32)
    state.prec = np.zeros((365, ymax - ymin + 1), dtype=np.float32)
    state.year = np.zeros((ymax - ymin + 1), dtype=np.float32)

    # Fill arrays
    for k, y in enumerate(range(ymin, ymax + 1)):
        IND = (temp_prec[:, 0] == y) & (temp_prec[:, 1] <= 365)
        state.prec[:, k] = temp_prec[IND, -1] * 365.0
        state.temp[:, k] = temp_prec[IND, -2]
        state.year[k] = y

    # Monthly option
    if cfg.processes.clim_spin_up.time_resolution == 12:
        II = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 364]
        state.prec = np.stack([
            np.mean(state.prec[II[i]:II[i + 1]], axis=0)
            for i in range(12)
        ])
        state.temp = np.stack([
            np.mean(state.temp[II[i]:II[i + 1]], axis=0)
            for i in range(12)
        ])

    # Select spin-up year
    if cfg.processes.clim_spin_up.spin_up_year is None:
        state.spin_idx = 0
    else:
        idx = np.where(state.year == cfg.processes.clim_spin_up.spin_up_year)[0]
        if len(idx) == 0:
            raise ValueError("Spin-up year not found in climate file")
        state.spin_idx = idx[0]

    # Initialize fields
    nt = len(state.temp)

    state.air_temp = tf.Variable(
        tf.zeros((nt, state.y.shape[0], state.x.shape[0])),
        dtype="float32",
    )

    state.precipitation = tf.Variable(
        tf.zeros((nt, state.y.shape[0], state.x.shape[0])),
        dtype="float32",
    )

    state.meanprec = tf.math.reduce_mean(state.precipitation, axis=0)
    state.meantemp = tf.math.reduce_mean(state.air_temp, axis=0)

    if "time" not in cfg.processes:
        raise ValueError("The 'time' module is required for the climate module.")

    state.tlast_clim_spinup = tf.Variable(cfg.processes.time.start)
    state.tcomp_clim_spinup = []


def update(cfg, state):

    if ((state.t - state.tlast_clim_spinup) >= cfg.processes.clim_spin_up.update_freq):

        if hasattr(state, "logger"):
            state.logger.info("update spin up climate at time : " + str(state.t.numpy()))

        state.tcomp_clim_spinup.append(time.time())

        # Vertical gradients
        dP = 0.00035
        dT = -0.00552

        # Always use the same year
        k = state.spin_idx

        PREC = tf.expand_dims(
            tf.expand_dims(state.prec[:, k], axis=-1), axis=-1
        )
        TEMP = tf.expand_dims(
            tf.expand_dims(state.temp[:, k], axis=-1), axis=-1
        )

        # Extend over glacier
        state.precipitation = tf.tile(PREC, (1, state.y.shape[0], state.x.shape[0]))
        state.air_temp = tf.tile(TEMP, (1, state.y.shape[0], state.x.shape[0]))

        # Altitude correction
        prec_corr_mult = 1 + dP * (state.usurf - state.zws)
        temp_corr_addi = dT * (state.usurf - state.zws)

        prec_corr_mult = tf.expand_dims(prec_corr_mult, axis=0)
        temp_corr_addi = tf.expand_dims(temp_corr_addi, axis=0)

        prec_corr_mult = tf.tile(prec_corr_mult, (len(state.prec), 1, 1))
        temp_corr_addi = tf.tile(temp_corr_addi, (len(state.temp), 1, 1))

        with tf.device('/CPU:0'):
            new_prec = tf.clip_by_value(
                state.precipitation * prec_corr_mult, 0, 1e10
            )

        state.precipitation = tf.identity(new_prec)
        state.air_temp = state.air_temp + temp_corr_addi

        state.meanprec = tf.math.reduce_mean(state.precipitation, axis=0)
        state.meantemp = tf.math.reduce_mean(state.air_temp, axis=0)

        state.tlast_clim_spinup.assign(state.t)

        state.tcomp_clim_spinup[-1] -= time.time()
        state.tcomp_clim_spinup[-1] *= -1


def finalize(cfg, state):
    pass
