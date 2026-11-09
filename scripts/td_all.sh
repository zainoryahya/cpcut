#!/usr/bin/env bash
# All time-dependent GPE runs of the paper (several CPU-hours). Results in results/td_*.json.
set -euo pipefail
cd "$(dirname "$0")/.."
P='{"dt_fac":0.5}'
# Table 5 and 6: g = 0 against stationary, and interacting runs (tuned RQF and Herwerth parabola)
for k in tuned_rqf parabola; do
  python scripts/td_runs.py "[[\"$k\",200,2.5,0,3000,$P]]" results/td_table5.json
  for g in 0 0.25 0.5 1.0; do
    python scripts/td_runs.py "[[\"$k\",150,1.0,$g,2000,$P],[\"$k\",175,1.0,$g,2000,$P]]" results/td_table6.json
  done
done
python scripts/td_runs.py '[["hard",200,2.5,0,3000,{}],["hard",200,1.0,0,3000,{}]]' results/td_table5.json
# delta = 100 nm, 2.5 mm/s
python scripts/td_runs.py "[[\"tuned_rqf\",100,2.5,0,3000,$P],[\"parabola\",100,2.5,0,3000,$P]]" results/td_table5.json
# convergence in grid spacing and time step (tuned RQF, delta = 200 nm, 2.5 mm/s)
python scripts/td_runs.py '[["tuned_rqf",200,2.5,0,3000,{"dt_fac":0.5,"pts_w":10}],["tuned_rqf",200,2.5,0,3000,{"dt_fac":0.5,"pts_w":40}],["tuned_rqf",200,2.5,0,3000,{"dt_fac":1.0}],["tuned_rqf",200,2.5,0,3000,{"dt_fac":0.5,"Lcap_fac":60}]]' results/td_conv.json
python scripts/td_mechanism.py                 # outgoing spurious wave with and without interactions
