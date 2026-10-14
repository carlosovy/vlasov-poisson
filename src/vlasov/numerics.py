
import numpy as np
from scipy.fft import fft, ifft, fftfreq

def compute_rho(f, dv):
    return dv * np.sum(f, axis=1)


def compute_efield(rho, dx):
    nx = len(rho)
    k = 2 * np.pi * fftfreq(nx, d=dx)

    rho_hat = fft(rho)
    e_hat = np.zeros_like(rho_hat)

    nonzero = k != 0
    e_hat[nonzero] = -1j * rho_hat[nonzero] / k[nonzero]

    return np.real(ifft(e_hat))


def advection_x(f, x, v, dt):
    # Your existing interpolation implementation
    ...

def advection_v(f, v, efield, dt):
    # Your existing interpolation implementation
    ...