"""Stationary 1D Schroedinger solver for quantum reflection.

psi'' = -k(x)^2 psi,  k^2 = (2m/hbar^2)(E - V).
Far field (x = ZF): psi = A exp(-ikx) + B exp(ikx), r = B/A.
Reference potential: incoming WKB wave at x0 (perfect absorption at the surface).
Cut-off potential: constant floor F below the transition; a plane wave exp(-iqx)
in the floor is exact (no reflection from the floor).

Speed-up: the outer region [delta, ZF] is common to every cut-off at the same
delta and velocity, so its real 2x2 transfer matrix is computed once and cached.
"""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp

ZF = 80000.0          # nm
RTOL, ATOL = 1e-11, 1e-13


def _transfer(k2, a, b, rtol=RTOL):
    """Real 2x2 matrix mapping (psi, psi') at a to (psi, psi') at b."""
    if b <= a:
        return np.eye(2)
    f = lambda x, y: [y[1], -k2(x) * y[0], y[3], -k2(x) * y[2]]
    s = solve_ivp(f, (a, b), [1.0, 0.0, 0.0, 1.0], method="DOP853", rtol=rtol, atol=ATOL)
    y = s.y[:, -1]
    return np.array([[y[0], y[2]], [y[1], y[3]]])


def _decompose(sp, v, psi, dpsi, zf):
    k = sp.k(v)
    A = 0.5 * (psi - dpsi / (1j * k)) * np.exp(1j * k * zf)
    B = 0.5 * (psi + dpsi / (1j * k)) * np.exp(-1j * k * zf)
    return B / A


@lru_cache(maxsize=4096)
def _outer(sp, d, v, zf=ZF):
    E = sp.energy(v)
    return _transfer(lambda x: sp.fac * (E - sp.V(x)), d, zf)


@lru_cache(maxsize=64)
def r_ref(sp, v, x0=0.125, zf=ZF, rtol=RTOL):
    """Reflection amplitude of the full potential, incoming WKB wave at x0 (nm)."""
    E = sp.energy(v)
    k2 = lambda x: sp.fac * (E - sp.V(x))
    k0 = np.sqrt(k2(x0))
    dk0 = -sp.fac * sp.dV(x0) / (2 * k0)
    y0 = [1 + 0j, -1j * k0 - dk0 / (2 * k0)]
    s = solve_ivp(lambda x, y: [y[1], -k2(x) * y[0]], (x0, zf), y0,
                  method="DOP853", rtol=rtol, atol=ATOL)
    return _decompose(sp, v, s.y[0, -1], s.y[1, -1], zf)


def r_cut(sp, cut, v, zf=ZF):
    """Reflection amplitude for a cut-off object (see cutoffs.py)."""
    E = sp.energy(v)
    q = np.sqrt(sp.fac * (E - cut.F))
    y = np.array([1 + 0j, -1j * q])                      # plane wave into the floor at xs
    if cut.xs < cut.d:
        y = _transfer(lambda x: sp.fac * (E - cut.Vin(x)), cut.xs, cut.d) @ y
    y = _outer(sp, float(cut.d), float(v), zf) @ y
    return _decompose(sp, v, y[0], y[1], zf)


def reflection(sp, v, cut=None, **kw):
    """Reflection probability |r|^2 (cut=None: reference potential)."""
    return abs(r_ref(sp, v, **kw) if cut is None else r_cut(sp, cut, v))**2
