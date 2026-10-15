"""Unit tests for density integration and the spectral field solver."""

import numpy as np

from vlasov.numerics import compute_efield, compute_rho


def test_compute_rho_constant_distribution():
    f = np.ones((4, 10))
    rho = compute_rho(f, dv=0.2)
    np.testing.assert_allclose(rho, np.full(4, 2.0))


def test_compute_efield_sinusoidal_density():
    nx = 64
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    dx = x[1] - x[0]
    rho = np.sin(x)

    efield = compute_efield(rho, dx)

    # dE/dx = rho - mean(rho), with mean(E) = 0.
    np.testing.assert_allclose(efield, -np.cos(x), rtol=1e-12, atol=1e-12)
