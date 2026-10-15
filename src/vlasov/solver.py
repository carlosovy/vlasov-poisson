
import numpy as np

from .numerics import (
    advection_x,
    advection_v,
    compute_rho,
    compute_efield,
    linear_interpolate,
)
from .diagnostics import electric_energy, total_rho, total_energy


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
    el_energy = np.empty(nsteps + 1)
    totrho = np.empty(nsteps + 1)
    totenergy = np.empty(nsteps + 1)

    for i in range(nsteps + 1):
        rho = compute_rho(f, grid.dv)
        efield = compute_efield(rho, grid.dx)
        el_energy[i] = electric_energy(efield, grid.dx)
        totrho[i] = total_rho(f,grid.dx, grid.dv)
        totenergy[i] = total_energy(f,efield,grid.v,grid.dx,grid.dv)

        if i < nsteps:
            f = step(f, grid, dt, interp_x, interp_v)

    return f, el_energy, totrho, totenergy
