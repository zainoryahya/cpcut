"""Random check of Theorem 1 / generalised Corollary 1 and of the tightness of gamma*,
using the cut-off data in normalised form (w = s = 1, V(delta) = 0, F = -u, D = -K)."""
import sys, pathlib, json
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cpcut.cutoffs import rqf_curve, gamma_star

t = np.linspace(0, 1, 20001)


def mono(u, K, a, b, g):
    S, _ = rqf_curve(-u, 0, 0, 0.0, 1.0, -K, 1.0, a, b, g)
    return bool(np.all(np.diff(S(t)) >= -1e-14))


def main(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    viol, ratios, tested = 0, [], 0
    for _ in range(n):
        u = 10**rng.uniform(-2, 0.5); K = 10**rng.uniform(-1, 2); b = 10**rng.uniform(-2, 2)
        amin = max(0.0, (1 - 3*u)/u) * b
        a = amin + 10**rng.uniform(-2, 2) * max(b, amin)
        gs = gamma_star(u, K, a, b)
        if not np.isfinite(gs):
            continue
        tested += 1
        for f in [0.1, 0.5, 0.99, 1.0]:
            if not mono(u, K, a, b, f*gs):
                viol += 1
        if len(ratios) < 300:      # tightness: largest monotone gamma by bisection
            lo, hi = gs, 50*gs
            if mono(u, K, a, b, hi):
                ratios.append(np.inf); continue
            for _ in range(40):
                mid = 0.5*(lo + hi)
                lo, hi = (mid, hi) if mono(u, K, a, b, mid) else (lo, mid)
            ratios.append(lo/gs)
    r = np.array(ratios); fin = r[np.isfinite(r)]
    print(f"tested {tested} cases (u from 0.01 to 3.2): violations {viol}")
    print(f"gamma_max/gamma*: median {np.median(fin):.3f}, 90% {np.percentile(fin,90):.3f}, max {fin.max():.3f}, unbounded {np.sum(~np.isfinite(r))} of {len(r)}")
    sub = [x for x in fin]
    json.dump(dict(tested=tested, violations=viol, median=float(np.median(fin)), p90=float(np.percentile(fin, 90)),
                   max=float(fin.max()), n_ratio=len(r)), open(pathlib.Path(__file__).resolve().parents[1]/"results/corollary.json", "w"))


if __name__ == "__main__":
    main()
