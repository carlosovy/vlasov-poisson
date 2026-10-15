import numpy as np

#Calculates several diagnostic quantities of interest from simulation step
#Useful for tracking time evolution of different quantities

def electric_energy(efield, dx):
    """Compute sum of logarithm of the electric-field amplitude.

    Parameters
    ----------
    efield : ndarray
        Electric field values with shape (nx,).
    dx : float
        Uniform spatial grid spacing.

    Returns
    -------
    float
        Logarithm of the spatial
        integral of the squared electric field.

    Notes
    -----
    The diagnostic is defined as

    .. math::

        D_E = \\frac{1}{2}\\ln\\left(
            \\Delta x \\sum_i E_i^2
        \\right).

    The logarithm is used to study exponential damping or
    growth of the electric field, which appears linear 
    with time, in the linear regime

    """
    return 0.5 *np.log( dx * np.sum(efield**2)) #Energy contained in the electric field, log is applied to see the linear slope in damping/instability

def total_rho(f,dx, dv):
    """Compute the total particle density integrated over phase space.

    Parameters
    ----------
    f : ndarray
        Distribution function with shape (nx, nv).
    dx : float
        Uniform spatial grid spacing.
    dv : float
        Uniform velocity grid spacing.

    Returns
    -------
    float
        Integral of the distribution function over the
        spatial and velocity domains.
    """
    return dx*dv * np.sum(f)

def total_energy(f,efield,v,dx,dv):
    """Compute the total kinetic and electric-field energy.

    Parameters
    ----------
    f : ndarray
        Distribution function with shape (nx, nv).
    efield : ndarray
        Electric field with shape (nx,).
    v : ndarray
        Velocity coordinates with shape (nv,).
    dx : float
        Spatial grid spacing.
    dv : float
        Velocity grid spacing.
    
    Returns
    -------
    float
        Sum of the kinetic and electric-field energies.
    
    Integrals are approximated using uniform-grid sums.
    """
    return 0.5*(dx*dv * np.sum(v**2*f) + dx * np.sum(efield**2)) #should be approximately conserved in time