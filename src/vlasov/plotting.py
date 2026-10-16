"""Reusable Matplotlib visualizations for dispersion and simulation results.

These helpers draw on a supplied Matplotlib Axes or create one when omitted.
Interactive widget controls remain in the notebooks.
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_dispersion(omega,strength,root=None,ax=None):
    """Plot a dispersion relation map.

    Parameters
    ----------
    omega : ndarray of complex
        Two-dimensional frequency grid, typically returned by
        ``dispersion_map``.
    strength : ndarray
        Two-dimensional residual strength values with the same shape as
        `omega`, typically ``-log10(abs(D))``.
    root : complex, optional
        Root to mark with a cyan star. No marker is drawn by default.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw. A new figure and axes are created if omitted.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the dispersion plot.
    mesh : matplotlib.collections.QuadMesh
        The pseudocolor mesh, useful for attaching a colorbar.

    Notes
    -----
    The displayed color scale is fixed to the range 0 to 6. A horizontal
    line marks zero imaginary frequency.
    """
    if ax is None: _,ax=plt.subplots()
    mesh=ax.pcolormesh(omega.real,omega.imag,strength,cmap='inferno',shading='auto',vmin=0,vmax=6)
    ax.axhline(0,color='white',alpha=.7,lw=.7)
    if root is not None: ax.plot(root.real,root.imag,'c*',ms=10)
    ax.set(xlabel=r'Re($\omega$)',ylabel=r'Im($\omega$)',title='Dispersion relation')
    return ax,mesh

def plot_distribution(v,distribution,ax=None):
    """Plot a distribution function as a function of velocity.

    Parameters
    ----------
    v : array_like
        One-dimensional velocity coordinates.
    distribution : array_like
        Distribution values corresponding to `v`.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. A new figure and axes are created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the distribution curve.
    """
    if ax is None: _,ax=plt.subplots()
    ax.plot(v,distribution)
    ax.set(xlabel='Velocity',ylabel='Distribution')
    return ax

def plot_growth_comparison(gamma_theory,gamma_simulation,ax=None):
    """Compare theoretical and simulated growth rates with a bar chart.

    Parameters
    ----------
    gamma_theory : float
        Growth rate obtained from linear theory.
    gamma_simulation : float
        Growth rate estimated from a simulation.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw. A new figure and axes are created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the growth-rate comparison.
    """
    if ax is None: _,ax=plt.subplots()
    ax.bar(['Linear theory','Simulation'],[gamma_theory,gamma_simulation])
    ax.set_ylabel(r'Growth rate $\gamma$')
    return ax
