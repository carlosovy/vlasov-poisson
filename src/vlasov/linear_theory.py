"""Linear dispersion analysis for a two-Maxwellian bump-on-tail distribution.

The background Maxwellian has unit temperature. The module evaluates the
complex dispersion relation, constructs a map of its inverse residual, and
searches numerically for growing modes.
"""
import numpy as np
from scipy.special import wofz
from scipy.optimize import root
from scipy.ndimage import maximum_filter

def plasma_dispersion(z):
    r"""Evaluate the plasma dispersion function using the Faddeeva function.

    Parameters
    ----------
    z : complex or array_like of complex
        Dimensionless complex argument.

    Returns
    -------
    complex or ndarray
        Plasma dispersion function evaluated at `z`.

    Notes
    -----
    The implemented convention is

    .. math::

        Z(z) = i\sqrt{\pi}\,w(z),

    where :math:`w` is the Faddeeva function from ``scipy.special.wofz``.
    """
    return 1j*np.sqrt(np.pi)*wofz(z)

def dispersion_relation(omega,k,eps,T0,v0):
    r"""Evaluate the two-Maxwellian electrostatic dispersion relation.

    Parameters
    ----------
    omega : complex or array_like of complex
        Complex wave frequency or array of frequencies.
    k : float
        Wavenumber of the mode.
    eps : float
        Amplitude of the second Maxwellian.
    T0 : float
        Temperature of the maxwellian, with the
        background temperature normalized to unity.
    v0 : float
        Velocity of the secondary Maxwellian.

    Returns
    -------
    complex or ndarray
        Dispersion-relation residual at `omega`. A mode satisfies D = 0.

    Raises
    ------
    ValueError
        If `k` is zero or `T0` is not positive.
    Notes
    -----
    Frequencies use the time-dependence convention
    :math:`\exp(-i\omega t)`, so positive imaginary frequency means growth.
    The returned residual is

    .. math::

        D = k^2 + (1-\epsilon)[1+z_1 Z(z_1)]
            + \frac{\epsilon}{T_0}[1+z_2 Z(z_2)],

    where :math:`z_1 = \omega/(\sqrt{2}|k|)` and
    :math:`z_2 = (\omega-kv_0)/(\sqrt{2T_0}|k|)`.
    """
    if k == 0 or T0 <= 0:
        raise ValueError('k must be nonzero and T0 positive')
    z1=omega/(np.sqrt(2)*abs(k))
    z2=(omega-k*v0)/(np.sqrt(2*T0)*abs(k))
    return (k*k + (1-eps)*(1+z1*plasma_dispersion(z1))
            + eps/T0*(1+z2*plasma_dispersion(z2)))

def dispersion_map(k,eps,T0,v0,omega_r,omega_i):
    """Evaluate the dispersion residual on a complex-frequency grid.
    Parameters
    ----------
    k : float
        Wavenumber of the mode.
    eps : float
        Amplitude of secondary Maxwellian.
    T0 : float
        Temperature of the Maxwellian.
    v0 : float
        Velocity of the Maxwellian.
    omega_r : array_like
        Real-frequency grid.
    omega_i : array_like
        Imaginary-frequency grid.

    Returns
    -------
    omega : ndarray of complex
        Complex-frequency mesh of shape ``(len(omega_i), len(omega_r))``.
    strength : ndarray
        Negative base-10 logarithm of the residual magnitude,
        ``-log10(abs(D))``, with the same shape as `omega`.

    Notes
    -----
    Larger `strength` values indicate smaller dispersion residuals.
    Exact zeros produce positive infinity; division/logarithm warnings
    are suppressed for the map calculation.
    """
    wr,wi=np.meshgrid(omega_r,omega_i)
    omega=wr+1j*wi
    with np.errstate(divide='ignore',invalid='ignore'):
        strength=-np.log10(np.abs(dispersion_relation(omega,k,eps,T0,v0)))
    return omega,strength

def dominant_mode(k,eps,T0,v0,omega_r=None,omega_i=None,n_candidates=25,
                  residual_tol=1e-7):
    """Find the fastest-growing unstable root among numerically found modes.

    Parameters
    ----------
    k : float
        Nonzero spatial wavenumber.
    eps : float
        Relative weight of the drifting Maxwellian.
    T0 : float
        Positive temperature of the drifting Maxwellian.
    v0 : float
        Drift velocity of the drifting Maxwellian.
    omega_r : array_like, optional
        Real-frequency search grid. By default, 240 points from -2 to 2.
    omega_i : array_like, optional
        Imaginary-frequency search grid. By default, 150 points from
        1e-4 to 1. The supplied grid should cover the desired growing modes.
    n_candidates : int, optional
        Maximum number of local residual peaks to refine. Default is 25.
    residual_tol : float, optional
        Maximum absolute dispersion residual accepted for a root.
        Default is 1e-7.

    Returns
    -------
    complex or None
        Found unstable root with the largest positive imaginary part,
        or ``None`` if no qualifying root is found.

    Notes
    -----
    Local maxima of ``-log10(abs(D))`` are selected from the search mesh
    and refined by ``scipy.optimize.root``. Candidate roots must have
    finite components, a positive imaginary part, and residual magnitude
    below `residual_tol`. Roots closer than 1e-5 are treated as duplicates.

    This heuristic search does not guarantee that all unstable roots are
    detected or that no unstable roots exist outside the search window.
    """
    if omega_r is None:
        omega_r=np.linspace(-2,2,240)
    if omega_i is None:
        omega_i=np.linspace(1e-4,1,150)
    omega,log_strength=dispersion_map(k,eps,T0,v0,omega_r,omega_i)
    good=np.isfinite(log_strength)
    scores=np.where(good,log_strength,-np.inf)
    local=(scores==maximum_filter(scores,size=5,mode='nearest')) & good
    coords=np.argwhere(local)
    coords=sorted(coords,key=lambda ij:scores[tuple(ij)],reverse=True)[:n_candidates]
    roots=[]
    def residual(w):
        d=dispersion_relation(w[0]+1j*w[1],k,eps,T0,v0)
        return [d.real,d.imag]
    for iy,ix in coords:
        guess=omega[iy,ix]
        sol=root(residual,[guess.real,guess.imag])
        w=complex(*sol.x)
        if (np.isfinite(w.real) and np.isfinite(w.imag) and w.imag>0
                and abs(dispersion_relation(w,k,eps,T0,v0))<residual_tol
                and not any(abs(w-r)<1e-5 for r in roots)):
            roots.append(w)
    return max(roots,key=lambda w:w.imag) if roots else None
