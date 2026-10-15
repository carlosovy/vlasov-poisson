
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

def bump_on_tail(grid, epsilon=0.001, kx=0.5, temperature=1.0,eps_bot=0.1,v_bot=3.8,t_bot=0.1):
    x = grid.x[:, None]
    v = grid.v[None, :]

    f1_eq=(1-eps_bot)*maxwellian(v, temperature) 
    f2_eq=eps_bot*maxwellian(v-v_bot, t_bot) 

    return  (f1_eq+f2_eq)* (1 + epsilon * np.cos(kx * x))