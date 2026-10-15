
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
    """Advance the distribution by one Strang-splitting time step.

    Parameters
    ----------
    f : ndarray
        Distribution function with shape (nx, nv).
    grid : Grid
        Spatial and velocity grid structure.
    dt : float
        Time step.
    interp_x, interp_v : callable, optional
        Interpolation functions used for spatial and velocity advection.

    Returns
    -------
    ndarray
        Updated distribution function with shape (nx, nv).
    
    Notes
    -----
    The Strang splitting scheme consists of three stages:
    
    1. Spatial advection over half a time step.
    2. Velocity advection over a full time step.
    3. Spatial advection over the remaining half time step.
    
    The electric field is computed after the first spatial
    advection stage.
    """
    f = advection_x(f, grid.x, grid.v, dt / 2, interp_x)

    #Need Poisson Solve for Efield to do velocity advection
    rho = compute_rho(f, grid.dv) 
    efield = compute_efield(rho, grid.dx)

    f = advection_v(f, grid.v, efield, dt, interp_v)
    f = advection_x(f, grid.x, grid.v, dt / 2, interp_x)

    return f


def run(f0, grid, dt, nsteps, interp_x=linear_interpolate, interp_v=linear_interpolate):
    f = f0.copy()
    fdistribu_evol = np.zeros((nsteps+1, grid.nx, grid.nv))
    el_energy = np.empty(nsteps + 1)
    totrho = np.empty(nsteps + 1)
    totenergy = np.empty(nsteps + 1)

    for i in range(nsteps + 1):
        rho = compute_rho(f, grid.dv)
        efield = compute_efield(rho, grid.dx)
        el_energy[i] = electric_energy(efield, grid.dx)
        totrho[i] = total_rho(f,grid.dx, grid.dv)
        totenergy[i] = total_energy(f,efield,grid.v,grid.dx,grid.dv)
        fdistribu_evol[i, :]=f

        if i < nsteps:
            f = step(f, grid, dt, interp_x, interp_v)

    return fdistribu_evol, el_energy, totrho, totenergy
