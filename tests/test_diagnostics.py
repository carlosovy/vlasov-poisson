"""Unit tests for phase-space diagnostics."""

import numpy as np

from vlasov.diagnostics import total_rho


def test_total_rho_constant_distribution():
    f = np.ones((4, 10))
    dx = 0.5
    dv = 0.2

    result = total_rho(f, dx, dv)

    np.testing.assert_allclose(result, 4 * 10 * dx * dv)
