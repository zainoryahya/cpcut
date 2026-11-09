# cpcut: shape-preserving C² cut-offs of the Casimir–Polder potential

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24070725.svg)](https://doi.org/10.5281/zenodo.24070725)

Code and data for

> Z. R. Yahya, M. N. A. Halif, *Shape-preserving C² cut-off of the Casimir–Polder potential for
> quantum-reflection simulations of Bose–Einstein condensates* (submitted to Computer Physics Communications).

The package builds truncated Casimir–Polder (CP) potentials for quantum-reflection solvers and
measures the artificial ("spurious") reflection each truncation causes:

* hard cut-off, the C¹ parabola of Herwerth et al. / Galiffi et al., C¹ cubic and C² polynomial
  quintic transitions, and the monotone C² rational quintic (RQF) transition of the paper, built from
  the C²-rational quintic function of Hussain, Hussain & Yameen (2018) with the monotonicity
  conditions of Theorem 1 / Corollary 1;
* a stationary scattering solver (DOP853) for the reflection amplitude, with the outer-region
  transfer matrix cached so that each further cut-off costs one short integration;
* a 1D time-dependent Gross–Pitaevskii solver (Strang split-step Fourier, cubic complex absorbing
  potentials).

## Install

```
pip install .              # the package (numpy, scipy)
pip install -r requirements.txt   # everything needed to reproduce the paper
```

Python ≥ 3.10. No compilation is needed. Quick check (a few seconds):

```
pip install pytest && pytest -q tests
```

## Use

```python
from cpcut import SPECIES, r_ref, r_cut, cutoffs as C
sp = SPECIES["Rb"]                       # Rb-87 on Si, parameters of Halif (2012)
d = 50.0                                 # cut-off position delta, nm
cut = C.rqf(sp, d, w=d, u=0.5, a=1e4, b=1.0, tg=0.93)   # tuned monotone C2 RQF (Table 2)
print(abs(r_cut(sp, cut, 1.0))**2, abs(r_ref(sp, 1.0))**2)   # R with cut-off, R untruncated, at 1 mm/s
print(C.is_monotone(cut))
```

To use the cut-off in another GPE code, evaluate `cut.Vin(x)` on `[cut.xs, cut.d]`, the CP
potential above `cut.d` and the constant `cut.F` below `cut.xs` (lengths in nm, energies in J).

## Reproduce the paper

`bash run_all.sh` runs every step in order and writes JSON files to `results/` and figures to
`figures/`. The stationary steps take minutes; the time-dependent runs take several CPU-hours.

| Script | Produces |
|---|---|
| `scripts/rqf_derive.py` | Symbolic checks: RQF interpolation conditions, reduction to the quintic Hermite polynomial, Bernstein coefficients of N (Eq. 9) |
| `scripts/check_corollary.py`, `scripts/tightness.py` | Random test of Theorem 1 / Corollary 1 and tightness of γ* |
| `scripts/reference.py` | Convergence of the reference amplitude in x0; threshold law R ≈ 1 − 4kβ₄ |
| `scripts/validate_thesis.py` | Table 1 (validation against Halif 2012) |
| `scripts/thesis_like.py` | Finite-difference grid study of the hard cut-off at δ = 0.3 μm |
| `scripts/tune.py` | Differential-evolution tuning of each cut-off family under equal-cost constraints (Table 2) |
| `scripts/stationary.py` | Figs. 1–2, Table 4, convergence orders, Appendix A check |
| `scripts/matched_u.py` | Table 3 (RQF against polynomial quintic at matched drop) |
| `scripts/cost.py` | Cost–accuracy data (Fig. 3) |
| `scripts/td_runs.py`, `scripts/td_all.sh` | Time-dependent GPE runs (Tables 5–6) |
| `scripts/td_mechanism.py` | Spurious outgoing wave with and without interactions (Section 7.3) |
| `results/td_earlier_*.json`, `results/td_parabola_g.json` | Earlier time-dependent runs (hard cut-off and parabola) made with the same solver before it was packaged; `td_all.sh` regenerates them |
| `scripts/make_figs.py` | Figures |
| `results/td_parabola200.json` | Parabola, δ = 200 nm, 2.5 mm/s (Table 5) |

Tables of the paper map to result files as follows: Table 1 `validate_thesis.json`;
Table 2 `tune_A_poly.json`, `tune_A_rqf.json`; Table 3 `matched_u.json`; Table 4 `stationary.json`
(`floor`); Table 5 `td_batch1.json`, `td_batch2.json`, `td_parabola200.json`, `td_earlier_*.json`;
Table 6 `td_batch1.json`, `td_parabola_g.json`, `td_earlier_2.json`.

## Cite

If you use this code, please cite the paper above and the archived software
(DOI: [10.5281/zenodo.24070725](https://doi.org/10.5281/zenodo.24070725)). Citation metadata are in `CITATION.cff`.

## Licence

MIT. See `LICENSE`.
