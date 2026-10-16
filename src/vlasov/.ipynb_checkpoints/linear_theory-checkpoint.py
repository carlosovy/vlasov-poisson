"""Two-Maxwellian bump-on-tail linear dispersion relation (background T=1)."""
import numpy as np
from scipy.special import wofz
from scipy.optimize import root
from scipy.ndimage import maximum_filter

def plasma_dispersion(z):
    return 1j*np.sqrt(np.pi)*wofz(z)

def dispersion_relation(omega,k,eps,T0,v0):
    """Evaluate D; roots D=0. Convention: exp(-i*omega*t)."""
    if k == 0 or T0 <= 0:
        raise ValueError('k must be nonzero and T0 positive')
    z1=omega/(np.sqrt(2)*abs(k))
    z2=(omega-k*v0)/(np.sqrt(2*T0)*abs(k))
    return (k*k + (1-eps)*(1+z1*plasma_dispersion(z1))
            + eps/T0*(1+z2*plasma_dispersion(z2)))

def dispersion_map(k,eps,T0,v0,omega_r,omega_i):
    """Return complex omega mesh and log10(1/|D|) for plotting."""
    wr,wi=np.meshgrid(omega_r,omega_i)
    omega=wr+1j*wi
    with np.errstate(divide='ignore',invalid='ignore'):
        strength=-np.log10(np.abs(dispersion_relation(omega,k,eps,T0,v0)))
    return omega,strength

def dominant_mode(k,eps,T0,v0,omega_r=None,omega_i=None,n_candidates=25,
                  residual_tol=1e-7):
    """Return largest-growth unstable root found within search window, else None.

    Candidate peaks on an upper-half-plane grid are refined with scipy.root.
    This is a numerical search, not a proof no unstable root exists.
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
