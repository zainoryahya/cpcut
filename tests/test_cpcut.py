"""Fast checks (about a minute): run with `pytest -q`."""
import numpy as np
from cpcut import SPECIES, r_ref, r_cut, cutoffs as C
from cpcut.cutoffs import rqf_curve, gamma_star

sp = SPECIES["Rb"]


def test_reference_reflection():
    # untruncated Rb-87 / Si at 1 mm/s (Section 2.2)
    assert abs(abs(r_ref(sp, 1.0))**2 - 5.4495e-3) < 2e-6


def test_hard_cutoff_at_300nm():
    # R / R_true = 4.1 at 1 mm/s (Section 6.1)
    ratio = abs(r_cut(sp, C.hard(sp, 300.0), 1.0))**2 / abs(r_ref(sp, 1.0))**2
    assert abs(ratio - 4.15) < 0.02


def test_tuned_rqf_is_monotone_and_accurate():
    cut = C.rqf(sp, 100.0, w=100.0, u=0.5, a=1e4, b=1.0, tg=0.93)
    assert C.is_monotone(cut)
    assert abs(abs(r_cut(sp, cut, 1.0) - r_ref(sp, 1.0)) - 9.37e-4) < 1e-5


def test_corollary1_random():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 5001)
    for _ in range(100):
        u = 10**rng.uniform(-2, 0.5); K = 10**rng.uniform(-1, 2); b = 10**rng.uniform(-2, 2)
        a = max(0.0, (1 - 3*u)/u) * b + 10**rng.uniform(-2, 2) * b
        g = gamma_star(u, K, a, b)
        if not np.isfinite(g):
            continue
        S, _ = rqf_curve(-u, 0, 0, 0.0, 1.0, -K, 1.0, a, b, g)
        assert np.all(np.diff(S(t)) >= -1e-14)


def test_polynomial_quintic_condition():
    # Corollary 2: monotone iff u >= (8+K)/20
    d, w = 100.0, 100.0
    K = C.K_of(sp, d, w); umin = (8 + K) / 20
    assert C.is_monotone(C.quintic(sp, d, w, umin + 0.01))
    assert not C.is_monotone(C.quintic(sp, d, w, umin - 0.05))
