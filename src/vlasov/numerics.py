
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
    alpha = dt * v

    f_new = np.stack([
        np.interp(x-alpha[j], x, f[:, j],period=x[-1]-x[0])
        for j in range(np.shape(f)[1])
    ], axis=1)
    return f_new

def advection_v(f, v, efield, dt):

    alpha = dt * efield

    f_new = np.stack([
        np.interp(v-alpha[i], v, f[i, :])
        for i in range(np.shape(f)[0])
    ], axis=0)
    return f_new