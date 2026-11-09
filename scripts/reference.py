"""Reference reflection amplitude: convergence in the WKB boundary position x0,
and an independent check of the solver against the analytic threshold law
|r|^2 = 1 - 4 k beta4 + O(k^2) for a pure -C4/x^4 potential."""
import json, sys, pathlib
from dataclasses import dataclass
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cpcut import SPECIES, Species, r_ref

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"


@dataclass(frozen=True)
class PureC4(Species):
    def V(self, x):
        return -self.c4 / x**4

    def dV(self, x):
        return 4 * self.c4 / x**5


def main():
    sp = SPECIES["Rb"]
    res = {"x0_conv": {}, "threshold": {}}
    for v in [0.5, 1.0, 2.5]:
        rs = {x0: r_ref(sp, v, x0=x0) for x0 in [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 10.0]}
        res["x0_conv"][str(v)] = {
            "R": abs(rs[0.125])**2,
            "dr_vs_x0_0.0625": {str(x0): float(abs(rs[x0] - rs[0.0625])) for x0 in rs}}
        print(f"v={v}: R_true={abs(rs[0.125])**2:.6e}  |r(x0)-r(0.0625nm)|:",
              {x0: f"{abs(rs[x0]-rs[0.0625]):.1e}" for x0 in rs}, flush=True)
    p = PureC4("C4only", sp.m, sp.C4, sp.lam)
    b4 = p.beta4
    rows = []
    for kb in [0.005, 0.01, 0.02, 0.04, 0.08]:
        k = kb / b4                                   # nm^-1
        v = k / np.sqrt(p.fac * p.energy(1.0))        # mm/s
        zf = 40 * 2 * np.pi / k
        R = abs(r_ref(p, v, x0=20.0, zf=zf))**2
        R2 = abs(r_ref(p, v, x0=20.0, zf=2 * zf))**2
        rows.append(dict(kbeta4=kb, v=v, R=R, ratio=(1 - R) / (4 * kb), dzf=abs(R - R2)))
        print(f"k beta4={kb}: v={v:.4f} mm/s  R={R:.6f}  (1-R)/(4 k beta4)={(1-R)/(4*kb):.4f}  zf-check {abs(R-R2):.1e}", flush=True)
    kb = np.array([r["kbeta4"] for r in rows]); y = np.array([r["ratio"] for r in rows])
    c = np.polyfit(kb[:4], y[:4], 1)
    print(f"linear extrapolation of (1-R)/(4k beta4) to k=0: {c[1]:.4f} (slope {c[0]:.3f})")
    res["threshold"] = dict(beta4_nm=b4, rows=rows, intercept=c[1], slope=c[0])
    OUT.mkdir(exist_ok=True)
    json.dump(res, open(OUT / "reference.json", "w"), indent=1)


if __name__ == "__main__":
    main()
