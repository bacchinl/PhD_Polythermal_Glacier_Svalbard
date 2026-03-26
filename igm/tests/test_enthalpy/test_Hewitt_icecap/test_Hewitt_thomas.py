#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Test Hewitt & Schoof (2017) ice cap enthalpy benchmark.

Implements the ice cap example from Section 4.3 of:
  Hewitt, I.J. and Schoof, C. (2017). Models for polythermal ice sheets and
  glaciers. The Cryosphere, 11, 541-551.

Runs 4 combinations matching Figs 7 (Ts=-10 °C) and 8 (Ts=-1 °C) of the paper,
each with and without ice-column drainage.  A paper-style figure (T cross-section
and basal melt rate) is saved for each case.

Validation checks (required for the test to pass):
  - No superheating (T ≤ 273.15 K, beta=0)
  - Temperature not below surface temperature
  - Non-negative water content
"""

import pytest

from .utils import run_ice_cap_test

pytestmark = pytest.mark.slow

_PARAMS = dict(Nx=501, Nz_E=500, dt=1.0, time_simu=5000.0, store_every=1000)
#_PARAMS = dict(Nx=101, Nz_E=100, dt=5.0, time_simu=5_000.0, store_every=100)


def test_hewitt_ice_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run all 4 Hewitt & Schoof ice cap cases (Figs 7 and 8)."""
    for T_s in [-10.0, -1.0]:
        for drain in [False, True]:
            run_ice_cap_test(monkeypatch, T_s=T_s, drain=drain, **_PARAMS)


print("END")
