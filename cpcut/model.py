"""Atom-surface model (Halif 2012 thesis parameters). Lengths in nm, energies in J."""
from dataclasses import dataclass
import numpy as np

HBAR = 1.054571817e-34


@dataclass(frozen=True)
class Species:
    name: str
    m: float        # kg
    C4: float       # J m^4
    lam: float      # m, lambda' = C4/C3

    @property
    def c4(self):   # J nm^4
        return self.C4 * 1e36

    @property
    def l(self):    # nm
        return self.lam * 1e9

    @property
    def fac(self):  # 2m/hbar^2 in J^-1 nm^-2
        return 2 * self.m / HBAR**2 * 1e-18

    @property
    def beta4(self):  # nm, sqrt(2 m C4)/hbar
        return np.sqrt(2 * self.m * self.C4) / HBAR * 1e9

    # V = -C4 / f,  f = x^3 (x + lam)
    def V(self, x):
        return -self.c4 / (x**3 * (x + self.l))

    def dV(self, x):
        f = x**4 + self.l * x**3
        f1 = 4 * x**3 + 3 * self.l * x**2
        return self.c4 * f1 / f**2

    def d2V(self, x):
        f = x**4 + self.l * x**3
        f1 = 4 * x**3 + 3 * self.l * x**2
        f2 = 12 * x**2 + 6 * self.l * x
        return self.c4 * (f2 * f - 2 * f1**2) / f**3

    def energy(self, v_mm_s):
        return 0.5 * self.m * (v_mm_s * 1e-3)**2

    def k(self, v_mm_s):  # nm^-1
        return np.sqrt(self.fac * self.energy(v_mm_s))


SPECIES = {
    "Rb": Species("Rb", 1.44e-25, 1.61e-55, 1.61e-55 / 1.69e-48),
    "Na": Species("Na", 3.81e-26, 9.1e-56, 589e-9 / (2 * np.pi)),
}
