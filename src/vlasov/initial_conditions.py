
import numpy as np

def maxwellian(v, temperature=1.0):
    return (
        np.exp(-v**2 / (2 * temperature))
        / np.sqrt(2 * np.pi * temperature)
    )

def landau_damping(grid, epsilon=0.001, kx=0.5, temperature=1.0):
    x = grid.x[:, None]
    v = grid.v[None, :]

    return maxwellian(v, temperature) * (
        1 + epsilon * np.cos(kx * x)
    )
