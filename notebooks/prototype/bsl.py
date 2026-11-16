#
#  B-Splines computation
#   see https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
#
# Written by P. Navaro (https://github.com/pnavaro)
#   see https://pnavaro.github.io/python-fortran/04.vlasov-poisson.html#semi-lagrangian-method
#   source: https://github.com/pnavaro/python-fortran
#
# MIT License
#
# Copyright (c) 2019 Pierre Navaro
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#---------------------------------------------------------------------

import numpy as np
from scipy.fft import fft, ifft
from scipy.fft import fftfreq  # noqa: F401


def bspline(p, j, x):
    """Return the value at x in [0,1[ of the B-spline with
    integer nodes of degree p with support starting at j.
    Implemented recursively using the de Boor's recursion formula"""
    assert (x >= 0.0) & (x <= 1.0)
    assert isinstance(p, int) & isinstance(j, int)
    if p == 0:
        if j == 0:
            return 1.0
        else:
            return 0.0
    else:
        w = (x - j) / p
        w1 = (x - j - 1) / p
    return w * bspline(p - 1, j, x) + (1 - w1) * bspline(p - 1, j + 1, x)


class BSpline:
    """Class to compute BSL advection of 1d function"""

    def __init__(self, p, xmin, xmax, ncells):
        assert p & 1 == 1  # check that p is odd
        self.p = p
        self.ncells = ncells
        # compute eigenvalues of degree p b-spline matrix
        self.modes = 2 * np.pi * np.arange(ncells) / ncells
        self.deltax = (xmax - xmin) / ncells

        self.eig_bspl = bspline(p, -(p + 1) // 2, 0.0)
        for j in range(1, (p + 1) // 2):
            self.eig_bspl += (
                bspline(p, j - (p + 1) // 2, 0.0) * 2 * np.cos(j * self.modes)
            )

        self.eigalpha = np.zeros(ncells, dtype=complex)

    def interpolate_disp(self, f, alpha):
        """compute the interpolating spline of degree p of odd degree
        of a function f on a periodic uniform mesh, at
        all points xi-alpha"""
        p = self.p
        assert np.size(f) == self.ncells
        # compute eigenvalues of cubic splines evaluated at displaced points
        ishift = np.floor(-alpha / self.deltax)
        beta = -ishift - alpha / self.deltax
        self.eigalpha.fill(0.0)
        for j in range(-(p - 1) // 2, (p + 1) // 2 + 1):
            self.eigalpha += bspline(p, j - (p + 1) // 2, beta) * np.exp(
                (ishift + j) * 1j * self.modes
            )

        # compute interpolating spline using fft and properties of circulant matrices
        return np.real(ifft(fft(f) * self.eigalpha / self.eig_bspl))
