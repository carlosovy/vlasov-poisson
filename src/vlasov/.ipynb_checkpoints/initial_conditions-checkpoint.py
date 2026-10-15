"""Initial distribution functions for Vlasov-Poisson simulations.

This module provides Maxwellian distributions and
perturbed initial conditions for studying Landau damping
and the bump-on-tail instability.
"""
import numpy as np

def maxwellian(v, temperature=1.0):
     """Evaluate a normalized one-dimensional Maxwellian distribution.
    Parameters
    ----------
    v : array_like
        Velocity coordinates at which to evaluate the distribution.
    temperature : float
        Normalized temperature, equal to the velocity
        variance. Default is 1.0.
        
    Returns
    -------
    ndarray
        Maxwellian distribution evaluated at `v`, with the same
        shape as the input array.
    """
    return (
        np.exp(-v**2 / (2 * temperature))
        / np.sqrt(2 * np.pi * temperature)
    )

def landau_damping(grid, epsilon=0.001, kx=0.5, temperature=1.0):
    """Generate a spatially perturbed Maxwellian distribution.
    Parameters
    ----------
    grid : Grid
        Spatial and velocity grid.
    epsilon : float, optional
        Amplitude of the introduced perturbation.
    kx : float, optional
        Wavenumber of the perturbation.
    temperature : float, optional
        Temperature of the Maxwellian distribution.

    Returns
    -------
    ndarray
        Initial distribution function with shape (nx, nv).

    Notes
    -----
    The initial distribution is a Maxwellian multiplied by
    a sinusoidal perturbation in space.

    The perturbation uses a cosine with wavenumber `kx`.
    """
    x = grid.x[:, None]
    v = grid.v[None, :]

    return maxwellian(v, temperature) * (
        1 + epsilon * np.cos(kx * x)
    )

def bump_on_tail(grid, epsilon=0.001, kx=0.5, temperature=1.0,eps_bot=0.1,v_bot=3.8,t_bot=0.1):
    """Generate a center Maxwellian and a second, displaced Maxwellian (bump), and perturb this distribution with cos(kx).
    Parameters
    ----------
    grid : Grid
        Spatial and velocity grid.
    epsilon : float, optional
        Amplitude of the introduced perturbation.
    kx : float, optional
        Wavenumber of the perturbation.
    temperature : float, optional
        Temperature of the main Maxwellian distribution.
    eps_bot : float, optional
        Amplitude of second Maxwellian (bump). Main maxwellian has 1-eps_bot amplitude
    v_bot : float, optional
        Position in velocity space of the bump
    t_bot: float, optional
        Temperature of the second Maxwellian

    Returns
    -------
    ndarray
        Initial distribution function with shape (nx, nv).
    """
    
    x = grid.x[:, None]
    v = grid.v[None, :]

    f1_eq=(1-eps_bot)*maxwellian(v, temperature) 
    f2_eq=eps_bot*maxwellian(v-v_bot, t_bot) 

    return  (f1_eq+f2_eq)* (1 + epsilon * np.cos(kx * x))