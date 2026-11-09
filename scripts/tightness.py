"""Tightness of gamma* (Cor. 1): largest monotone gamma / gamma*, split by u >= 1/3 and u < 1/3."""
import sys, pathlib, json
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from check_corollary import mono
from cpcut.cutoffs import gamma_star


def main(n=200, seed=2):
    rng = np.random.default_rng(seed)
    out = {}
    for lab, (ulo, uhi) in {"u>=1/3": (np.log10(1/3), 0.5), "u<1/3": (-2, np.log10(1/3))}.items():
        rat = []
        while len(rat) < n:
            u = 10**rng.uniform(ulo, uhi); K = 10**rng.uniform(-1, 2); b = 10**rng.uniform(-2, 2)
            amin = max(0.0, (1 - 3*u)/u) * b
            a = amin + 10**rng.uniform(-2, 2) * max(b, amin)
            gs = gamma_star(u, K, a, b)
            if not np.isfinite(gs):
                continue
            lo, hi = gs, 100*gs
            if mono(u, K, a, b, hi):
                rat.append(np.inf); continue
            for _ in range(30):
                mid = np.sqrt(lo*hi)
                lo, hi = (mid, hi) if mono(u, K, a, b, mid) else (lo, mid)
            rat.append(lo/gs)
        r = np.array(rat); f = r[np.isfinite(r)]
        out[lab] = dict(median=float(np.median(f)), p90=float(np.percentile(f, 90)), max=float(f.max()),
                        frac_within_5pct=float(np.mean(r <= 1.05)), n=len(r))
        print(lab, out[lab], flush=True)
    json.dump(out, open(pathlib.Path(__file__).resolve().parents[1] / "results/tightness.json", "w"), indent=1)


if __name__ == "__main__":
    main()
