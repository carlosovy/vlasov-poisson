
import numpy as np

from .numerics import (
    advection_x,
    advection_v,
    compute_rho,
    compute_efield,
    linear_interpolate,
)
from .diagnostics import electric_energy


def step(f, grid, dt, interp_x=linear_interpolate, interp_v=linear_interpolate):
    """One Strang-splitting time step."""
    f = advection_x(f, grid.x, grid.v, dt / 2, interp_x)

    rho = compute_rho(f, grid.dv)
    efield = compute_efield(rho, grid.dx)

    f = advection_v(f, grid.v, efield, dt, interp_v)
    f = advection_x(f, grid.x, grid.v, dt / 2, interp_x)

    return f


def run(f0, grid, dt, nsteps, interp_x=linear_interpolate, interp_v=linear_interpolate):
    f = f0.copy()
    energy = np.empty(nsteps + 1)

    for i in range(nsteps + 1):
        rho = compute_rho(f, grid.dv)
        efield = compute_efield(rho, grid.dx)
        energy[i] = electric_energy(efield, grid.dx)

        if i < nsteps:
            f = step(f, grid, dt, interp_x, interp_v)

    return f, energy
