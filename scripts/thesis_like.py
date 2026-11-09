"""Thesis-like discretisation of the hard cut-off at delta = 0.3 um.

The time-dependent solver of Halif (2012) uses a second-order (3-point) finite-difference
Laplacian with a grid step set by the de Broglie wavelength at the cut-off, and a linear
negative imaginary potential (NIP) -i V_IM (delta - x) below the cut-off. Its monochromatic,
time-converged limit is the discrete stationary equation
   psi_{n+1} = (2 - dx^2 k_n^2) psi_n - psi_{n-1},   k_n^2 = (2m/hbar^2)(E - V_n + i W_n),
solved here by outward recursion from deep inside the absorber, with the far field split
into the discrete plane waves exp(+-i kappa x), cos(kappa dx) = 1 - dx^2 k^2/2.
"""
import json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cpcut import SPECIES, r_ref, r_cut, cutoffs as C

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
sp = SPECIES["Rb"]
V_IM = 3.6e-26 * 1e-9        # J/nm  (thesis value 3.6e-26 J/m)


def R_fd(d, v, dx, vim=V_IM, depth=None, zf=80000.0, hard=True):
    E = sp.energy(v)
    kf = np.sqrt(sp.fac * (E - sp.V(d)))
    if depth is None:   # absorber depth giving int Im k dx ~ 25
        depth = np.sqrt(50 * kf / (sp.fac * vim))
    x = np.arange(d - depth, zf + dx, dx)
    Vx = np.where(x >= d, sp.V(np.maximum(x, d)), sp.V(d))
    W = np.where(x < d, vim * (d - x), 0.0)
    a = 2 - dx * dx * sp.fac * (E - Vx + 1j * W)
    p0, p1 = 0.0 + 0j, 1e-300 + 0j
    for n in range(1, len(x)):
        p0, p1 = p1, a[n - 1] * p1 - p0
        if abs(p1) > 1e250:
            p0, p1 = p0 * 1e-250, p1 * 1e-250
    # p0 = psi at x[-2], p1 = psi at x[-1];  psi_n = A e^{-i kap x_n} + B e^{i kap x_n}
    k = sp.k(v)
    kap = np.arccos(1 - dx * dx * k * k / 2) / dx
    x0, x1 = x[-2], x[-1]
    M = np.array([[np.exp(-1j*kap*x0), np.exp(1j*kap*x0)], [np.exp(-1j*kap*x1), np.exp(1j*kap*x1)]])
    A, B = np.linalg.solve(M, [p0, p1])
    return abs(B / A)**2, kf


def main():
    d = 300.0
    vs = [0.5, 1.0, 2.5]
    thesis = {0.5: 0.08, 1.0: 0.045, 2.5: 0.02}
    out = {"delta": d, "rows": [], "R_true": [abs(r_ref(sp, v))**2 for v in vs],
           "R_hard_ideal": [abs(r_cut(sp, C.hard(sp, d), v))**2 for v in vs], "thesis": thesis}
    lam_f = 2 * np.pi / np.sqrt(sp.fac * (sp.energy(1.0) - sp.V(d)))
    print(f"lambda_f = {lam_f:.0f} nm; ideal hard: " + " ".join(f"{x:.4f}" for x in out["R_hard_ideal"]),
          " true: " + " ".join(f"{x:.4f}" for x in out["R_true"]))
    for N in [2, 3, 4, 5, 6, 8, 10, 15, 20, 40, 80]:
        dx = lam_f / N
        Rs = [R_fd(d, v, dx)[0] for v in vs]
        out["rows"].append(dict(N=N, dx=dx, R=Rs))
        print(f"dx = lambda_f/{N:<3d} ({dx:6.1f} nm): R = " + " ".join(f"{x:.4f}" for x in Rs)
              + "   ratio to ideal: " + " ".join(f"{a/b:.2f}" for a, b in zip(Rs, out["R_hard_ideal"])), flush=True)
    # absorber check: fine grid, NIP slope x10 and /10
    for f in [0.1, 10.0]:
        Rs = [R_fd(d, v, lam_f / 80, vim=V_IM * f)[0] for v in vs]
        out[f"nip_x{f}"] = Rs
        print(f"NIP slope x{f}, dx=lambda_f/80: " + " ".join(f"{x:.4f}" for x in Rs))
    OUT.mkdir(exist_ok=True)
    json.dump(out, open(OUT / "thesis_like.json", "w"), indent=1)


if __name__ == "__main__":
    main()
