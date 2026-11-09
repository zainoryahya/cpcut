"""Monotone RQF against the polynomial quintic at the same delta, w and drop u,
so the floor and width are identical and only the shape differs (audit item: the
earlier monotone/non-monotone comparison was confounded with u)."""
import json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cpcut import SPECIES, r_ref, r_cut, cutoffs as C

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
sp = SPECIES["Rb"]
vs = [0.5, 1.0, 2.5]
t = np.linspace(0, 1, 4001)


def shape_stats(cut):
    y = cut.S(t); dy = np.diff(y)
    drop = abs(y[-1] - y[0])
    return float((dy < 0).mean()), float(max(0, y[0] - y.min(), y.max() - y[-1]) / drop)


def main():
    out = {}
    for d, w in [(100, 100), (100, 200), (200, 200)]:
        K = C.K_of(sp, d, w); umin = (8 + K) / 20
        rows = []
        for u in np.linspace(0.10, 0.95, 18):
            q = C.quintic(sp, d, w, u)
            a = max(1.0, 2 * 2.0 * (1 - 3*u) / u)      # alpha/beta = 2x the Cor. 1 bound when u < 1/3
            r = C.rqf(sp, d, w, u, a, 2.0)
            fq, xq = shape_stats(q)
            row = dict(u=float(u), q_mono=C.is_monotone(q), r_mono=C.is_monotone(r), q_frac=fq, q_exc=xq,
                       q_err=[abs(abs(r_cut(sp, q, v))**2 / abs(r_ref(sp, v))**2 - 1) for v in vs],
                       r_err=[abs(abs(r_cut(sp, r, v))**2 / abs(r_ref(sp, v))**2 - 1) for v in vs],
                       q_amp=abs(r_cut(sp, q, 1.0) - r_ref(sp, 1.0)), r_amp=abs(r_cut(sp, r, 1.0) - r_ref(sp, 1.0)))
            rows.append(row)
        nm = [r for r in rows if not r["q_mono"]]
        print(f"delta={d} w={w} K={K:.1f} u_min={umin:.2f}: quintic non-monotone for {len(nm)}/18; RQF monotone for "
              f"{sum(r['r_mono'] for r in rows)}/18")
        for i, v in enumerate(vs):
            qe = [r["q_err"][i] for r in nm]; re = [r["r_err"][i] for r in nm]
            wins = sum(a < b for a, b in zip(re, qe))
            print(f"   v={v}: over non-monotone-quintic drops: quintic median/max {np.median(qe):.3f}/{max(qe):.3f}, "
                  f"RQF {np.median(re):.3f}/{max(re):.3f}; RQF better in {wins}/{len(nm)}")
        ra = [r["q_amp"] / r["r_amp"] for r in nm]
        print(f"   amplitude ratio quintic/RQF at 1 mm/s: median {np.median(ra):.2f}, min {min(ra):.2f}, max {max(ra):.2f}")
        out[f"{d},{w}"] = dict(K=K, umin=umin, rows=rows)
    json.dump(out, open(OUT / "matched_u.json", "w"), indent=1)


if __name__ == "__main__":
    main()
