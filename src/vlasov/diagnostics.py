import numpy as np

def electric_energy(efield, dx):
    return 0.5 *np.log( dx * np.sum(efield**2))