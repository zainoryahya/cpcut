"""Cut-off potentials below x = delta (all match V at delta).

Each builder returns a Cut with: d (delta), xs (start of the transition, the
floor is x < xs), F (floor value), Vin (potential on [xs, d]).
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class Cut:
    d: float
    xs: float
    F: float
    Vin: Callable
    name: str = ""
    S: Callable = None      # transition curve on t in [0,1] (if any)


def hard(sp, d):
    Vd = sp.V(d)
    return Cut(d, d, Vd, lambda x: Vd, "hard")


def parabola(sp, d):
    """C1 parabola of Herwerth et al. (2013) / Galiffi et al. (2017): vertex at x = 0,
    value and slope of V matched at delta, constant V(delta) - s delta/2 below 0."""
    s, Vd = sp.dV(d), sp.V(d)
    Vin = lambda x: Vd + s * (x * x - d * d) / (2 * d)
    return Cut(d, 0.0, Vd - s * d / 2, Vin, "parabola", lambda t: Vin(t * d))


def cubic(sp, d, w, u):
    """C1 cubic Hermite transition of width w, floor drop u*w*V'(delta)."""
    s, Vd = sp.dV(d), sp.V(d)
    F = Vd - u * w * s

    def S(t):
        return F * (2*t**3 - 3*t**2 + 1) + Vd * (-2*t**3 + 3*t**2) + s * w * (t**3 - t**2)
    return Cut(d, d - w, F, lambda x: S((x - (d - w)) / w), "cubic", S)


def gamma_star(u, K, a, b):
    """Corollary 1: largest gamma for which P2, P3, P4 > 0 (min over the three bounds)."""
    c = K + 4 - 6 * u
    if c <= 0:
        return np.inf
    g2 = (2*u*a + (6*u - 2)*b) / c
    den = c * b - (12*u - 4) * a
    g3 = np.inf if den <= 0 else ((6*u - 2)*b*b + 6*u*a*b) / den
    L = (18*u - 2)*a + (18*u - 4 + K)*b
    g4 = (L + np.sqrt(L*L + 8*c*b*b)) / (2*c)
    return min(g2, g3, g4)


def rqf_curve(fi, di, Di, fj, dj, Dj, h, a, b, g):
    """C2 rational quintic of Hussain, Hussain & Yameen (2018), eq. (1), on t in [0,1]."""
    A = [a*fi, (3*a + b)*fi + a*h*di, (3*a + 3*b + g)*fi + (2*a + b)*h*di + 0.5*a*h*h*Di,
         (3*g + 3*b + a)*fj - (2*g + b)*h*dj + 0.5*g*h*h*Dj, (3*g + b)*fj - g*h*dj, g*fj]
    C = [1, 5, 10, 10, 5, 1]

    def S(t):
        p = sum((1 - t)**(5 - k) * t**k * A[k] for k in range(6))
        return p / (a*(1 - t)**2 + b*(1 - t)*t + g*t*t)
    return S, A


def rqf(sp, d, w, u, a=1.0, b=2.0, g=None, tg=0.9):
    """RQF transition of width w on [delta-w, delta]: C2 match to V at delta, flat
    (f=f'=f''=0 relative) floor F = V(delta) - u w V'(delta) at delta-w.
    g=None -> g = tg * gamma*(u, K, a, b). a=1, b=2, g=1 gives the polynomial quintic."""
    s, s2, Vd = sp.dV(d), sp.d2V(d), sp.V(d)
    K = -w * s2 / s
    F = Vd - u * w * s
    if g is None:
        g = tg * gamma_star(u, K, a, b)
    S, _ = rqf_curve(F, 0, 0, Vd, s, s2, w, a, b, g)
    return Cut(d, d - w, F, lambda x: S((x - (d - w)) / w), "rqf", S)


def quintic(sp, d, w, u):
    """C2 polynomial quintic Hermite (RQF with alpha=1, beta=2, gamma=1)."""
    c = rqf(sp, d, w, u, 1.0, 2.0, 1.0)
    c.name = "quintic"
    return c


def K_of(sp, d, w):
    return -w * sp.d2V(d) / sp.dV(d)


def is_monotone(cut, n=4001):
    if cut.S is None:
        return True
    y = cut.S(np.linspace(0, 1, n))
    return bool(np.all(np.diff(y) >= 0))
