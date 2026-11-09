"""1D time-dependent Gross-Pitaevskii solver (Strang split-step Fourier) for quantum
reflection from a truncated CP potential. x in nm, SI otherwise. The surface is at
small x; atoms come from the right.

Absorbers: a cubic complex absorbing potential (CAP) in the floor, of length
Lcap_fac floor wavelengths and strength giving int Im k dx ~ 40, starting two floor
wavelengths below the transition; and a weaker cubic CAP on the far right, sized for
the incident wavenumber, which removes the reflected packet.
R = 1 - P_abs(left) - P_inner(x < x_s) at the end of the run.
Grid: dx = min(lambda_f/4, w/pts_w) for smooth cut-offs (lambda_f/dx_lam for the hard
cut-off); time step dt = hbar/(4|V_min|)/dt_fac. lambda_f is the wavelength at the deepest
point V_min of the truncated potential (the floor, for a monotone transition).
"""
import numpy as np
import scipy.fft as sfft
from .model import HBAR


def _cap(x, x_end, L, W0):
    z = np.clip((x_end - x) / L, 0, None)
    return W0 * z**3


def setup(sp, cut, v, sigma=3000.0, x0=None, dt_fac=1.0, g_over_E=0.0, Lcap_fac=30,
          pts_w=20, dx_lam=None, fd=False, extra_right=0.0):
    m = sp.m
    E = sp.energy(v)
    F = cut.F
    w = cut.d - cut.xs if cut.xs < cut.d else np.inf
    Vmin = F if cut.S is None else min(F, float(np.min(cut.S(np.linspace(0, 1, 801)))))
    kf = np.sqrt(2 * m * (E - Vmin)) / HBAR * 1e-9          # largest local wavenumber
    lamf = 2 * np.pi / kf
    Lc = Lcap_fac * lamf
    W0 = 2 * 20 * 4 * HBAR**2 * (kf * 1e9) / (m * Lc * 1e-9)
    xcap_end = cut.xs - 2 * lamf
    xL = xcap_end - Lc - 5 * lamf
    if x0 is None:
        x0 = 6 * sigma + 3000.0
    k0 = m * v * 1e-3 / HBAR * 1e-9
    LR = 10000.0
    xR = x0 + 6 * sigma + extra_right + LR
    kmax = 2.0 * max(kf, k0 + 8 / (2 * sigma))
    dx = min(np.pi / kmax, w / pts_w)
    if dx_lam:
        dx = min(dx, lamf / dx_lam)
    N = int(2**np.ceil(np.log2((xR - xL) / dx)))
    x = xL + np.arange(N) * (xR - xL) / N
    dx = x[1] - x[0]
    Vr = np.full(N, F)
    a = (x >= cut.xs) & (x < cut.d)
    b = x >= cut.d
    if a.any():
        Vr[a] = cut.Vin(x[a])
    Vr[b] = sp.V(x[b])
    W0R = 10 * 4 * HBAR**2 * (k0 * 1e9) / (m * LR * 1e-9)
    W = _cap(x, xcap_end, Lc, W0) + _cap(-x, -(xR - LR), LR, W0R)
    dt = 0.25 * HBAR / abs(Vmin) / dt_fac
    dt = min(dt, 0.05 * (sigma * 1e-9) / (v * 1e-3))
    psi = (2 * np.pi * sigma**2)**-0.25 * np.exp(-(x - x0)**2 / (4 * sigma**2) - 1j * k0 * x)
    psi /= np.sqrt(np.sum(abs(psi)**2) * dx)
    g = g_over_E * E * np.sqrt(2 * np.pi) * sigma        # g max|psi0|^2 = g_over_E * E
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    if fd:                                               # 3-point finite-difference kinetic symbol
        k = np.sqrt(2 - 2 * np.cos(k * dx)) / dx
    return dict(sp=sp, x=x, dx=dx, N=N, Vr=Vr, W=W, dt=dt, psi=psi, k=k, g=g, E=E, x0=x0, F=F,
                kf=kf, xcap_end=xcap_end, xR=xR, LR=LR, sigma=sigma, v=v, Lc=Lc)


def default_tmax(S):
    return 1.5 * (S["x0"] + 6 * S["sigma"]) * 1e-9 / (0.35 * S["v"] * 1e-3)


def cost(S, t_max=None):
    """Work estimate N log2 N x number of steps."""
    t_max = default_tmax(S) if t_max is None else t_max
    return S["N"] * np.log2(S["N"]) * t_max / S["dt"]


def run(S, x_s=2500.0, t_max=None, tol=2e-3, workers=1, t_stop=None):
    m = S["sp"].m
    x, dx, dt, psi, k = S["x"], S["dx"], S["dt"], S["psi"].copy(), S["k"]
    T = np.exp(-1j * HBAR * (k * 1e9)**2 / (2 * m) * dt)
    halfW = np.exp(-S["W"] * dt / (2 * HBAR))
    left = x < S["xcap_end"]
    inner = x < x_s
    loss = (1 - halfW**2) * left
    P_left, t, v = 0.0, 0.0, S["v"] * 1e-3
    t_max = default_tmax(S) if t_max is None else t_max
    nsteps = int(t_max / dt)
    chk = max(1, nsteps // 400)
    g = S["g"]
    U = np.exp(-1j * S["Vr"] * dt / (2 * HBAR)) * halfW
    for n in range(nsteps):
        if g != 0.0:
            psi *= np.exp(-1j * g * (psi.real**2 + psi.imag**2) * dt / (2 * HBAR))
        P_left += np.dot(psi.real**2 + psi.imag**2, loss) * dx
        psi *= U
        psi = sfft.ifft(T * sfft.fft(psi, workers=workers), workers=workers)
        P_left += np.dot(psi.real**2 + psi.imag**2, loss) * dx
        psi *= U
        if g != 0.0:
            psi *= np.exp(-1j * g * (psi.real**2 + psi.imag**2) * dt / (2 * HBAR))
        t += dt
        if t_stop is not None and t >= t_stop:
            Nin = np.sum(abs(psi[inner])**2) * dx
            R = 1 - P_left - Nin
            break
        if t_stop is None and (n % chk == 0 or n == nsteps - 1):
            Nin = np.sum(abs(psi[inner])**2) * dx
            R = 1 - P_left - Nin
            if t > (S["x0"] - x_s) * 1e-9 / v * 1.2 and Nin < tol * max(R, 1e-12):
                break
    return dict(R=R, inner=Nin, t=t, steps=n + 1, psi=psi)


def R_packet(sp, cut, v, sigma, n=24):
    """Stationary R averaged over the packet's momentum distribution."""
    from .solver import r_cut, r_ref
    k0 = sp.m * v * 1e-3 / HBAR * 1e-9
    z, wq = np.polynomial.hermite_e.hermegauss(n)
    wq = wq / wq.sum()
    ks = k0 + z / (2 * sigma)
    vs = HBAR * ks * 1e9 / sp.m * 1e3
    f = (lambda vv: abs(r_ref(sp, vv))**2) if cut is None else (lambda vv: abs(r_cut(sp, cut, vv))**2)
    return float(sum(w * f(vv) for w, vv in zip(wq, vs) if vv > 0))
