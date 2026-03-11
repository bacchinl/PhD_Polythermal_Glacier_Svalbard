#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf



@tf.function
def compute_heat_refreeze(
    frac_refreezing : tf.Tensor,
    omega: tf.Tensor,
    L_ice: tf.Tensor,
) -> tf.Tensor:
    """
    TensorFlow function to compute the heat released by the refreezing

    Args:
        frac_refreezing : Fraction of the percolating that refreeze
        omega : liquid water content.
        L_ice : Latent heat of fusion for ice (J kg^-1).

    Returns:
        Heat released by refreezing (J kg^-1 yr^-1).
    """
  
    return frac_refreezing*omega*L_ice 
