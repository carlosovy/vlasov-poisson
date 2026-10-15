
from dataclasses import dataclass
import numpy as np

#Dataclass for space and velocity grid
#Properties like minimum, maximum, number of points, or the array itself can be accessed
@dataclass(frozen=True) #Frozen means its attributes cant be changed after creation
class Grid:
    xmin: float
    xmax: float
    nx: int
    vmin: float
    vmax: float
    nv: int

    @property
    def x(self):
        return np.linspace(
            self.xmin, self.xmax, self.nx, endpoint=False
        )

    @property
    def v(self):
        return np.linspace(
            self.vmin, self.vmax, self.nv, endpoint=False
        )

    @property
    def dx(self):
        return (self.xmax - self.xmin) / self.nx

    @property
    def dv(self):
        return (self.vmax - self.vmin) / self.nv
