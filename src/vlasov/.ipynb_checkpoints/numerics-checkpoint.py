"""Numerical methods for the one-dimensional Vlasov-Poisson solver.

This module provides methods for computing particle density,
solving for the electric field using Fourier transforms, and
performing semi-Lagrangian advection in phase space.

Both linear and cubic spline interpolation are supported.
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import CubicSpline

def compute_rho(f, dv):
    """Compute the spatial density by integrating over velocity.

    Parameters
    ----------
    f : ndarray
        Distribution function with shape (nx, nv).
    dv : float
        Uniform velocity grid spacing.

    Returns
    -------
    ndarray
        Spatial density with shape (nx,).

    Notes
    -----
    The velocity integral is approximated by a discrete sum:

    .. math::

        \\rho(x_i) = \\Delta v \\sum_j f(x_i, v_j).

    The velocity dimension corresponds to axis 1 of `f`.
    """
    return dv * np.sum(f, axis=1)


def compute_efield(rho, dx):
    nx = len(rho)
    k = 2 * np.pi * fftfreq(nx, d=dx)

    rho_hat = fft(rho)
    e_hat = np.zeros_like(rho_hat)

    nonzero = k != 0
    e_hat[nonzero] = -1j * rho_hat[nonzero] / k[nonzero]

    return np.real(ifft(e_hat))


def linear_interpolate(grid, values, points, periodic=False):
    """Linear interpolation; clamp at endpoints if nonperiodic."""
    if periodic:
        period = (grid[1] - grid[0]) * len(grid)
        return np.interp(points, grid, values, period=period)
    return np.interp(points, grid, values)



def cubic_interpolate(grid, values, points, periodic=False):
    """Cubic spline interpolation with periodic or clamped boundaries."""
    if periodic:
        period = (grid[1] - grid[0]) * len(grid)
        # Include the periodic endpoint required by scipy's CubicSpline.
        x = np.append(grid, grid[0] + period)
        y = np.append(values, values[0])
        spline = CubicSpline(x, y, bc_type="periodic")
        return spline(points)
    spline = CubicSpline(grid, values, bc_type="natural")
    return spline(np.clip(points, grid[0], grid[-1]))




def advection_x(f, x, v, dt, interpolate=linear_interpolate):
    """Advect the distribution function along space.
    The advection is performed using a semi-Lagrangian method,
    evaluating the distribution at the departure points.

    Parameters
    ----------
    f : ndarray
        Distribution function with shape (nx, nv).
    x : ndarray
        Spatial grid coordinates with shape (nx,).
    v : ndarray
        Velocity grid coordinates with shape (nv,).
    dt : float
        Time step used for advection.
    interpolate : callable, optional
        Interpolation function used to evaluate the distribution
        at departure points. Defaults to linear interpolation.

    Returns
    -------
    ndarray
        Advected distribution function with shape (nx, nv).

    Notes
    -----
    The departure points are computed as
    .. math::

        x_d = x - v \Delta t.

    Periodic boundary conditions are imposed in the spatial
    direction.
    """

    alpha = dt * v

    f_new = np.stack([
        interpolate(x, f[:, j], x - alpha[j], periodic=True)
        for j in range(f.shape[1])
    ], axis=1)
    return f_new

def advection_v(f, v, efield, dt, interpolate=linear_interpolate):
    """Advect the distribution function along the velocity direction.

    The advection is performed using a semi-Lagrangian method,
    evaluating the distribution at the departure points.

    Parameters
    ----------
    f : ndarray
        Distribution function with shape (nx, nv).
    v : ndarray
        Velocity grid coordinates with shape (nv,).
    efield : ndarray
        Electric field array with shape (nx,).
    dt : float
        Time step used for advection.
    interpolate : callable, optional
        Interpolation function used to evaluate the distribution
        at departure points. Defaults to linear interpolation.

    Returns
    -------
    ndarray
        Advected distribution function with shape (nx, nv).

    Notes
    -----
    The departure points are computed as

    .. math::

        x_d = x - v \Delta t.

    Periodic boundary conditions are imposed in the spatial
    direction.
    """

    alpha = dt * efield

    f_new = np.stack([
        interpolate(v, f[i, :], v - alpha[i], periodic=False)
        for i in range(f.shape[0])
    ], axis=0)
    return f_new

