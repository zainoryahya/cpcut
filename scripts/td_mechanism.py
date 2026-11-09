"""Direct look at the spurious reflected wave with and without interactions.
Runs the parabola and the tuned RQF at delta = 150 nm, v = 1 mm/s, g = 0 and g = 0.5 E,
stops all runs at the same time t_s, and takes the k > 0 (outgoing) part phi of the wave in
x > x_s. phi_s = phi_parab - phi_rqf is the difference of the two spurious waves, and
phi_int = phi_rqf(g) - phi_rqf(0) is the reflected wave added by interactions."""
import json, sys, pathlib, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cpcut import SPECIES, td
from stationary import families

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
sp = SPECIES["Rb"]


def outgoing(S, psi, x_s=2500.0):
    x, dx = S["x"], S["dx"]
    win = 0.5 * (1 + np.tanh((x - x_s) / 300.0)) * (x < S["xR"] - S["LR"])
    f = np.fft.fft(psi * win)
    k = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)
    f[k <= 0] = 0
    return np.fft.ifft(f)


def main(d=150.0, v=1.0, sigma=2000.0, t_s=0.035, gs=(0.0, 0.5)):
    F = families()
    phi, info = {}, {}
    for kind in ["parabola", "tuned_rqf"]:
        for g in gs:
            S = td.setup(sp, F[kind](d), v, sigma=sigma, g_over_E=g, dt_fac=0.5, extra_right=20000.0)
            t0 = time.time()
            r = td.run(S, t_stop=t_s)
            p = outgoing(S, r["psi"])
            phi[(kind, g)] = (p, S["dx"], S["x"])
            info[f"{kind},{g}"] = dict(N=S["N"], P_out=float(np.sum(abs(p)**2) * S["dx"]), R_so_far=r["R"], secs=time.time() - t0)
            print(kind, g, info[f"{kind},{g}"], flush=True)
    # all runs share the grid (same floor and width -> same N and dx)
    dx = phi[("parabola", 0.0)][1]
    ip = lambda a, b: np.vdot(a, b) * dx
    out = {"runs": info}
    ps0 = phi[("parabola", 0.0)][0] - phi[("tuned_rqf", 0.0)][0]
    for g in gs:
        pr = phi[("tuned_rqf", g)][0]; ps = phi[("parabola", g)][0] - pr
        pint = pr - phi[("tuned_rqf", 0.0)][0]
        row = dict(norm_s=float(np.sqrt(ip(ps, ps).real)), norm_r=float(np.sqrt(ip(pr, pr).real)),
                   dP=float((ip(phi[("parabola", g)][0], phi[("parabola", g)][0]) - ip(pr, pr)).real),
                   overlap_s_s0=float(abs(ip(ps0, ps)) / np.sqrt(ip(ps0, ps0).real * ip(ps, ps).real)))
        if g != 0.0:
            row["norm_int"] = float(np.sqrt(ip(pint, pint).real))
            row["coh_int_s"] = float(abs(ip(pint, ps)) / np.sqrt(ip(pint, pint).real * ip(ps, ps).real))
            row["cross_int"] = float(2 * ip(pint, ps).real)
        out[str(g)] = row
        print(g, row, flush=True)
    json.dump(out, open(RES / "td_mechanism.json", "w"), indent=1)


if __name__ == "__main__":
    main()
