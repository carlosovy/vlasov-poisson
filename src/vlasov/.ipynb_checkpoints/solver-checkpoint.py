
import numpy as np

from .numerics import (
    advection_x,
    advection_v,
    compute_rho,
    compute_efield,
)
from .diagnostics import electric_energy


def step(f, grid, dt):
    """One Strang-splitting time step."""
    f = advection_x(f, grid.x, grid.v, dt / 2)

    rho = compute_rho(f, grid.dv)
    efield = compute_efield(rho, grid.dx)

    f = advection_v(f, grid.v, efield, dt)
    f = advection_x(f, grid.x, grid.v, dt / 2)

    return f


def run(f0, grid, dt, nsteps):
    f = f0.copy()
    energy = np.empty(nsteps + 1)

    for i in range(nsteps + 1):
        rho = compute_rho(f, grid.dv)
        efield = compute_efield(rho, grid.dx)
        energy[i] = electric_energy(efield, grid.dx)

        if i < nsteps:
            f = step(f, grid, dt)

    return f, energy
