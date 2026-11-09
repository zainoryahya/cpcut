"""Time-dependent GPE runs. Usage:
   python scripts/td_runs.py '<json list of [kind, delta, v, g_over_E, sigma, {options}]>' out.json
kind: hard | parabola | default_rqf | tuned_rqf | tuned_cubic | tuned_quintic
options: dt_fac, pts_w, dx_lam, Lcap_fac, fd (passed to cpcut.td.setup)."""
import json, sys, os, time, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cpcut import SPECIES, td
from stationary import families

sp = SPECIES["Rb"]


def main(jobs, outf):
    F = families()
    res = json.load(open(outf)) if os.path.exists(outf) else []
    for kind, d, v, g, sig, opt in jobs:
        cut = F[kind](float(d))
        kw = dict(sigma=sig, g_over_E=g)
        if kind == "hard":
            kw["dx_lam"] = 45
        kw.update(opt)
        S = td.setup(sp, cut, v, **kw)
        t0 = time.time()
        r = td.run(S)
        Rs = td.R_packet(sp, cut, v, sig) if g == 0 else None
        row = dict(kind=kind, delta=d, v=v, g=g, sigma=sig, opt=opt, R=r["R"], inner=r["inner"], N=S["N"],
                   dx=S["dx"], dt=S["dt"], steps=r["steps"], cost=td.cost(S), secs=time.time() - t0, R_stat=Rs)
        res.append(row)
        json.dump(res, open(outf, "w"), indent=1)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]), sys.argv[2])
