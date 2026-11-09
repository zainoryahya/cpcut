"""Cost-accuracy of each cut-off in the time-dependent solver: work N log2 N x steps of a
converged run (grid and time-step rules of cpcut.td) against the truncation error
|R/R_true - 1| from the stationary solver, with delta as the parameter along each curve."""
import json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cpcut import SPECIES, r_ref, r_cut, td
from stationary import families

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
sp = SPECIES["Rb"]
vs = [0.5, 1.0, 2.5]


def main(sigma=2000.0):
    F = families()
    ds = np.geomspace(20, 300, 13)
    out = {"delta": ds.tolist(), "sigma": sigma, "fam": {}, "sec_per_unit": 1.2e-9,
           "amp_1pct": {str(v): 0.005 * abs(r_ref(sp, v)) for v in vs}}
    for name, b in F.items():
        rows = []
        for d in ds:
            c = b(float(d))
            S = td.setup(sp, c, 1.0, sigma=sigma, dx_lam=45 if name == "hard" else None,
                         dt_fac=1.0 if name == "hard" else 0.5)      # rules used in the runs
            err = [abs(abs(r_cut(sp, c, v))**2 / abs(r_ref(sp, v))**2 - 1) for v in vs]
            amp = abs(r_cut(sp, c, 1.0) - r_ref(sp, 1.0))
            rows.append(dict(d=float(d), N=S["N"], dx=S["dx"], dt=S["dt"], cost=td.cost(S), err=err, amp=amp))
        out["fam"][name] = rows
        print(name, " ".join(f"{r['d']:.0f}:{r['cost']*1.2e-9/3600:.2g}h/{r['amp']:.1e}" for r in rows), flush=True)
    json.dump(out, open(RES / "cost.json", "w"), indent=1)


if __name__ == "__main__":
    main()
