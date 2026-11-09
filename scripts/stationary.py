"""Stationary results: R/R_true and spurious amplitude versus delta for every cut-off
(Figs. 1-2, Table 2), convergence orders, 1% limits, the Appendix A prefactor check,
and the floor-depth table (Table 4)."""
import json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cpcut import SPECIES, r_ref, r_cut, cutoffs as C
import tune

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
sp = SPECIES["Rb"]
vs = [0.5, 1.0, 2.5]


def families():
    f = {"hard": lambda d: C.hard(sp, d), "parabola": lambda d: C.parabola(sp, d),
         "default_rqf": lambda d: C.rqf(sp, d, 2*d, 0.5)}
    for tag, suffix in [("A_poly", ""), ("A_rqf", ""), ("BM_rqf", "_narrow")]:
        p = RES / f"tune_{tag}.json"
        if not p.exists():
            continue
        for fam, rec in json.load(open(p))["fams"].items():
            if fam == "cubic_m":
                continue
            x = rec["best"]["x"]
            f["tuned_" + fam + suffix] = (lambda d, fam=fam, x=x: tune.build(fam, x, d))
    return f


def main():
    F = families()
    out = {"v": vs, "R_true": [abs(r_ref(sp, v))**2 for v in vs]}
    dh = np.geomspace(20, 3000, 34); da = np.geomspace(20, 600, 15)
    out["delta_R"] = dh.tolist(); out["delta_amp"] = da.tolist()
    out["ratio"], out["amp"], out["orders"], out["lim1"], out["max50"], out["lam_min"] = {}, {}, {}, {}, {}, {}
    for name, b in F.items():
        out["ratio"][name] = {str(v): [abs(r_cut(sp, b(d), v))**2 / out["R_true"][i] for d in dh] for i, v in enumerate(vs)}
        out["amp"][name] = {str(v): [abs(r_cut(sp, b(d), v) - r_ref(sp, v)) for d in da] for v in vs}
        out["lam_min"][name] = [2*np.pi/np.sqrt(sp.fac*(sp.energy(1.0) - b(d).F)) for d in da]
        e = np.array(out["amp"][name]["1.0"]); sel = da <= 150
        out["orders"][name] = float(np.polyfit(np.log(da[sel]), np.log(e[sel]), 1)[0])
        lim, mx = [], []
        for v in vs:
            r = np.abs(np.array(out["ratio"][name][str(v)]) - 1)
            bad = np.nonzero(r > 0.01)[0]
            lim.append(float(dh[bad[0]]) if len(bad) else None)
            mx.append(float(r[dh <= 50].max()))
        out["lim1"][name], out["max50"][name] = lim, mx
        a = out["amp"][name]
        spread = max(abs(np.array(a["0.5"]) / np.array(a["1.0"]) - 1).max(), abs(np.array(a["2.5"]) / np.array(a["1.0"]) - 1)[da <= 150].max())
        print(f"{name:13s} order {out['orders'][name]:.2f}; |dr| at 20,30,50,100,200,300 nm: "
              + " ".join(f"{np.interp(np.log(x), np.log(da), e):.2e}" for x in [20, 30, 50, 100, 200, 300])
              + f"; R ratio at 20 nm: {', '.join(f'{out['ratio'][name][str(v)][0]:.4f}' for v in vs)}"
              + f"; first delta with >1% error: {lim}; max error delta<=50: {[f'{x:.4f}' for x in mx]}; speed spread {spread:.3f}", flush=True)
    i3 = int(np.argmin(abs(dh - 300)))
    print("hard at 0.3 um R/R_true:", [round(out["ratio"]["hard"][str(v)][i3], 2) for v in vs])
    # Appendix A prefactor check at v = 1 mm/s
    E = sp.energy(1.0); k = lambda V: np.sqrt(sp.fac * (E - V))
    pred = lambda D, j, kk: (sp.fac/2) * abs(D) / (2**(j+1) * kk**(j+2))
    d3V = lambda x, e=1e-4: (sp.d2V(x*(1+e)) - sp.d2V(x*(1-e))) / (2*e*x)
    pre = []
    for i, d in enumerate(da):
        if d > 160:
            break
        s, s2, s3, Vd = sp.dV(d), sp.d2V(d), d3V(d), sp.V(d)
        h = pred(s, 1, k(Vd)) / out["amp"]["hard"]["1.0"][i]
        p = pred(s2 - s/d, 2, k(Vd)) / out["amp"]["parabola"]["1.0"][i]
        c = C.rqf(sp, d, 2*d, 0.5); hh = 1e-3; w = 2*d
        y = c.S(np.array([1-3*hh, 1-2*hh, 1-hh, 1.0])); S3 = (y[3]-3*y[2]+3*y[1]-y[0]) / hh**3 / w**3
        r = pred(S3 - s3, 3, k(Vd)) / out["amp"]["default_rqf"]["1.0"][i]
        pre.append(dict(d=float(d), hard=h, parab=p, rqf=r))
        print(f"  prefactor check delta={d:6.1f}: pred/meas hard {h:.3f}, parabola {p:.3f}, default RQF {r:.3f}")
    out["prefactor"] = pre
    # floor-depth table
    t = np.linspace(0, 1, 4001); fl = []
    for d, w in [(100, 400), (50, 400), (50, 800)]:
        K = C.K_of(sp, d, w); umin = (8 + K) / 20
        q = C.quintic(sp, d, w, umin + 0.01); r = C.rqf(sp, d, w, 0.5)
        eq = [abs(abs(r_cut(sp, q, v))**2 / out["R_true"][i] - 1) for i, v in enumerate(vs)]
        er = [abs(abs(r_cut(sp, r, v))**2 / out["R_true"][i] - 1) for i, v in enumerate(vs)]
        row = dict(d=d, w=w, K=K, umin=umin, Fq=q.F/sp.V(d), Fr=r.F/sp.V(d), refine=float(np.sqrt(q.F/r.F)),
                   eq=eq, er=er, mono_q=C.is_monotone(q), mono_r=C.is_monotone(r))
        fl.append(row)
        print(f"  floor d={d} w={w} K={K:.1f} umin={umin:.2f} Fq/V={row['Fq']:.1f} Fr/V={row['Fr']:.1f} refine {row['refine']:.2f} "
              f"err quintic {np.round(eq, 4)} rqf {np.round(er, 4)} mono {row['mono_q']},{row['mono_r']}")
    out["floor"] = fl
    json.dump(out, open(RES / "stationary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
