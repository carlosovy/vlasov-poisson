import numpy as np

def electric_energy(efield, dx):
    return 0.5 *np.log( dx * np.sum(efield**2))

def total_rho(f,dx, dv):
    return dx*dv * np.sum(f)

def total_energy(f,efield,v,dx,dv):
    return 0.5*(dx*dv * np.sum(v**2*f) + dx * np.sum(efield**2))