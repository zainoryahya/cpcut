#!/usr/bin/env bash
# Reproduce every number and figure of the paper. Stationary steps take minutes;
# the time-dependent runs (step 6) take several CPU-hours in total.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/rqf_derive.py                 # symbolic checks of the RQF and its Bernstein coefficients
python scripts/check_corollary.py            # random test of Theorem 1 / Corollary 1
python scripts/tightness.py                  # tightness of gamma*
python scripts/reference.py                  # reference amplitude: x0 convergence and threshold law
python scripts/validate_thesis.py            # Table 1
python scripts/thesis_like.py                # finite-difference grid study at delta = 0.3 um
python scripts/tune.py A_poly cubic,cubic_m,quintic 3   # equal-cost tuning, C1 and C2 polynomial families
python scripts/tune.py A_rqf rqf 5           # equal-cost tuning, monotone RQF
python scripts/tune.py BM_rqf rqf 3          # RQF, width relaxed to w >= 0.3 delta, tuned over delta = 30-100 nm
python scripts/stationary.py                 # Figs. 1-2, Tables 2 and 4, Appendix A check
python scripts/matched_u.py                  # Table 3
python scripts/cost.py                       # cost-accuracy data (Fig. 4)
bash scripts/td_all.sh                       # time-dependent GPE runs (Tables 5-6)
python scripts/make_figs.py                  # figures
