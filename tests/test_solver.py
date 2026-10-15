"""Short integration and regression tests for the Vlasov-Poisson solver."""

import numpy as np
import pytest

from vlasov.config import Grid
from vlasov.initial_conditions import bump_on_tail, landau_damping
from vlasov.numerics import cubic_interpolate, linear_interpolate
from vlasov.solver import run


@pytest.fixture
def small_grid():
    """A grid that keeps the solver tests fast."""
    return Grid(xmin=0, xmax=4 * np.pi, nx=32, vmin=-6, vmax=6, nv=32)


@pytest.mark.parametrize("initial_condition", [landau_damping, bump_on_tail])
@pytest.mark.parametrize("interpolate", [linear_interpolate, cubic_interpolate])
def test_simulation_runs(small_grid, initial_condition, interpolate):
    """Both physical setups should run with both interpolation schemes."""
    nsteps = 5
    f0 = initial_condition(small_grid, epsilon=0.001, kx=0.5)

    f_history, energy, total_density, total_energy = run(
        f0, small_grid, dt=0.6, nsteps=nsteps,
        interp_x=interpolate, interp_v=interpolate,
    )

    assert f_history.shape == (nsteps + 1, small_grid.nx, small_grid.nv)
    assert energy.shape == (nsteps + 1,)
    assert total_density.shape == (nsteps + 1,)
    assert total_energy.shape == (nsteps + 1,)

    for output in (f_history, energy, total_density, total_energy):
        assert np.all(np.isfinite(output))

    np.testing.assert_array_equal(f_history[0], f0)


def test_run_preserves_initial_condition(small_grid):
    f0 = landau_damping(small_grid)
    original = f0.copy()

    run(f0, small_grid, dt=0.6, nsteps=3)

    np.testing.assert_array_equal(f0, original)


def test_particle_number_approximately_conserved(small_grid):
    """Check short-run mass conservation with a pragmatic starter tolerance."""
    f0 = landau_damping(small_grid, epsilon=0.001)
    _, _, total_density, _ = run(f0, small_grid, dt=0.2, nsteps=5)

    relative_error = np.max(np.abs(total_density - total_density[0])) / abs(total_density[0])
    # Tighten this empirically after choosing numerical accuracy requirements.
    assert relative_error < 1e-2
