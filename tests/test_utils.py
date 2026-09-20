import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import target_kernels

KAPPA_PHI = 2.0
KAPPA_DELTA = 3.0
NU_DELTA = 0.5
DT = 1 / 12
N = 20

STATIONARY_VAR = 0.5 * NU_DELTA**2 / KAPPA_DELTA  # nu^2 / (2 kappa)


def _kernels(
    kappa_phi=KAPPA_PHI, kappa_delta=KAPPA_DELTA, nu_delta=NU_DELTA, dt=DT, n=N
):
    return target_kernels(kappa_phi, kappa_delta, nu_delta, dt, n)


def test_reversion_kernels_match_closed_form():
    """delta_reversion[p] = exp(-kappa_delta * p * dt), phi_reversion likewise."""
    k = _kernels()
    offset = DT * np.arange(N + 1)
    assert np.allclose(k.delta_reversion, np.exp(-KAPPA_DELTA * offset))
    assert np.allclose(k.phi_reversion, np.exp(-KAPPA_PHI * offset))


def test_log_delta_var_matches_closed_form():
    k = _kernels()
    offset = DT * np.arange(N + 1)
    expected = STATIONARY_VAR * (1 - np.exp(-2 * KAPPA_DELTA * offset))
    assert np.allclose(k.log_delta_var, expected)


def test_inner_kernel_shape_and_sign():
    k = _kernels()
    assert np.all(np.triu(k.inner_kernel) == 0.0)
    assert np.all(np.diag(k.inner_kernel) == 0.0)
    assert np.all(k.inner_kernel[np.tril_indices(N + 1, -1)] > 0)
