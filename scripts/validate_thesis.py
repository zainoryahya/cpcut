"""R versus hard cut-off position (validation against Halif 2012, Fig. 4.6) and the Na checks."""
import json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cpcut import SPECIES, r_ref, r_cut, cutoffs as C

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"


def main():
    rb, na = SPECIES["Rb"], SPECIES["Na"]
    vs = [0.5, 1.0, 2.5]
    ds = np.arange(300, 3001, 100)
    out = {"v": vs, "delta_nm": ds.tolist(), "R_true": [abs(r_ref(rb, v))**2 for v in vs], "R_hard": {}}
    for v in vs:
        out["R_hard"][str(v)] = [abs(r_cut(rb, C.hard(rb, float(d)), v))**2 for d in ds]
        row = out["R_hard"][str(v)]
        i = int(np.argmax(row)); dips = [int(ds[j]) for j in range(1, len(ds)-1) if row[j] < row[j-1] and row[j] < row[j+1]]
        print(f"Rb v={v}: R_true={out['R_true'][vs.index(v)]:.3e}  max {row[i]:.3f} at {ds[i]} nm, dips at {dips}")
    out["Na"] = {str(v): dict(R_true=abs(r_ref(na, v))**2, R_hard_150=abs(r_cut(na, C.hard(na, 150.0), v))**2) for v in [1.0, 2.5]}
    for v, d in out["Na"].items():
        print(f"Na v={v}: R_true={d['R_true']:.4f} hard(0.15um)={d['R_hard_150']:.4f} ratio {d['R_hard_150']/d['R_true']:.3f}")
    OUT.mkdir(exist_ok=True)
    json.dump(out, open(OUT / "validate_thesis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
