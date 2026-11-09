"""Differential-evolution tuning of cut-off families under equal-cost constraints.

Cost constraint (same as or cheaper than the Herwerth/Galiffi parabola in both solvers):
  floor drop  u*w <= delta/2, and no point of the transition below the parabola's floor
              (-> lambda_min and time step no smaller)
  width       w   >= delta     (transition no narrower -> same grid-spacing rule w/20)
Families (all match V, V' at delta; C2 ones also V''):
  cubic      C1 cubic Hermite (w, u)            contains every free-width parabola (u = 1/2)
  cubic_m    same, restricted to monotone (u >= 1/3, Fritsch-Carlson)
  quintic    C2 polynomial quintic (w, u)       monotone iff u >= (8+K)/20 (Cor. 2)
  rqf        C2 monotone RQF (w, u, alpha, beta, t = gamma/gamma*), Cor. 1 for any u > 0
Objective: spurious amplitude |r_cut - r_true| at delta = 100 nm, v = 1 mm/s.
"""
import json, sys, pathlib, time
import numpy as np
from scipy.optimize import differential_evolution
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cpcut import SPECIES, r_ref, r_cut, cutoffs as C

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
sp = SPECIES["Rb"]
D0, V0 = 100.0, 1.0


def build(fam, p, d, wmin=1.0):
    """p are box coordinates in [0,1]-ish ranges; returns a Cut."""
    if fam in ("cubic", "cubic_m", "quintic"):
        wf, phi = p
        w = wf * d
        u = phi * 0.5 / wf                       # u w = phi * delta/2
        if fam == "cubic_m":                     # monotone cubic: u >= 1/3 (Fritsch-Carlson), so w <= 1.5 delta
            u = 1/3 + phi * (0.5 / wf - 1/3)
        return (C.cubic if fam.startswith("cubic") else C.quintic)(sp, d, w, u)
    wf, phi, la, lb, t = p
    w = wf * d
    u = phi * 0.5 / wf
    a, b = 10**la, 10**lb
    if 2*u*a + (6*u - 2)*b <= 0:                 # P2(0) > 0  <=>  alpha/beta > (1-3u)/u
        return None
    K = C.K_of(sp, d, w)
    g = t * C.gamma_star(u, K, a, b)
    if not (np.isfinite(g) and g > 0):
        return None
    return C.rqf(sp, d, w, u, a, b, g)


def bounds(fam, wmin):
    if fam == "cubic_m":
        return [(wmin, 1.5), (0.0, 1.0)]
    if fam in ("cubic", "quintic"):
        return [(wmin, 12.0), (0.005, 1.0)]
    return [(wmin, 12.0), (0.005, 1.0), (-2.0, 2.0), (-2.0, 2.0), (0.02, 1.0)]


TT = np.linspace(0, 1, 801)


def too_deep(c):
    """Cost constraint on the deepest point, not just the floor: a non-monotone transition
    may dip below its floor, and the deepest point sets lambda_min and the time step."""
    return c.S is not None and c.S(TT).min() < sp.V(c.d) - 0.5 * sp.dV(c.d) * c.d * (1 + 1e-9)


def err(fam, p, d=D0, v=V0):
    c = build(fam, p, d)
    if c is None or too_deep(c):
        return 1.0
    try:
        e = abs(r_cut(sp, c, v) - r_ref(sp, v))
        return e if np.isfinite(e) else 1.0
    except Exception:
        return 1.0


def err_multi(fam, p, ds=(30.0, 50.0, 100.0)):
    """Mean log10 spurious amplitude over several offsets (guards against tuning a
    cancellation that holds at one delta only)."""
    return float(np.mean([np.log10(err(fam, p, d)) for d in ds]))


def tune(fam, seed, wmin=1.0, popsize=15, maxiter=60, multi=False):
    obj = (lambda p: err_multi(fam, p)) if multi else (lambda p: err(fam, p))
    res = differential_evolution(obj, bounds(fam, wmin), popsize=popsize,
                                 maxiter=maxiter, seed=seed, polish=False, tol=1e-12, mutation=(0.5, 1.0))
    return res.x, res.fun, res.nfev


def main(wmin=1.0, seeds=range(5), tag="A", fams=("cubic", "cubic_m", "quintic", "rqf"), multi=False):
    out = {"wmin": wmin, "fams": {}}
    ds, vs = [20, 30, 50, 100, 150, 200, 300], [0.5, 1.0, 2.5]
    out["parabola"] = {str(d): [abs(r_cut(sp, C.parabola(sp, d), v) - r_ref(sp, v)) for v in vs] for d in ds}
    out["default_rqf"] = {str(d): [abs(r_cut(sp, C.rqf(sp, d, 2*d, 0.5), v) - r_ref(sp, v)) for v in vs] for d in ds}
    for fam in fams:
        runs = []
        for s in seeds:
            t0 = time.time()
            x, f, n = tune(fam, s, wmin, multi=multi)
            f = err(fam, x)
            c = build(fam, x, D0)
            runs.append(dict(seed=s, x=x.tolist(), f=f, nfev=n, mono=(C.is_monotone(c) if c is not None else None), time=time.time() - t0))
            print(f"[{tag}] {fam:8s} seed {s}: |dr|={f:.3e} x={np.round(x,3)} mono={runs[-1]['mono']} ({n} evals, {time.time()-t0:.0f}s)", flush=True)
        best = min(runs, key=lambda r: r["f"])
        tr = {str(d): [err(fam, best["x"], d, v) for v in vs] for d in ds}
        mono = {str(d): (C.is_monotone(build(fam, best['x'], d)) if build(fam, best['x'], d) is not None else None) for d in ds}
        out["fams"][fam] = dict(runs=runs, best=best, transfer=tr, mono=mono)
        print(f"   best {fam}: transfer v=1: " + " ".join(f"{d}:{tr[str(d)][1]:.2e}" for d in ds), flush=True)
    print("   parabola  v=1: " + " ".join(f"{d}:{out['parabola'][str(d)][1]:.2e}" for d in ds))
    print("   default   v=1: " + " ".join(f"{d}:{out['default_rqf'][str(d)][1]:.2e}" for d in ds))
    OUT.mkdir(exist_ok=True)
    json.dump(out, open(OUT / f"tune_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "A"
    fams = sys.argv[2].split(",") if len(sys.argv) > 2 else ("cubic", "cubic_m", "quintic", "rqf")
    nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    main(wmin={"A": 1.0, "B": 0.3}[tag[0]], tag=tag, fams=fams, seeds=range(nseed), multi="M" in tag)
