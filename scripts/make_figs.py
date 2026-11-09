"""Figures 1-4 from results/*.json."""
import json, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cpcut import SPECIES, cutoffs as C
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "mathtext.fontset": "cm"})
RED, GREY, BLUE, LBLUE, GREEN, ORANGE = "#c8423b", "#8a8984", "#2a78d6", "#8fbbe9", "#3a9a5b", "#d98a1f"
STY = {"hard": (RED, "o", "-", "hard cut-off"),
       "parabola": (GREY, "^", "-", "C$^1$ Herwerth parabola"),
       "tuned_cubic": (ORANGE, "v", ":", "C$^1$ cubic, tuned"),
       "tuned_quintic": (GREEN, "x", "--", "C$^2$ quintic, tuned (non-monotone)"),
       "default_rqf": (LBLUE, "s", "-", "monotone C$^2$ RQF, default"),
       "tuned_rqf": (BLUE, "D", "-", "monotone C$^2$ RQF, tuned"),
       "tuned_rqf_narrow": (BLUE, "d", ":", "monotone C$^2$ RQF, tuned, $w\\geq0.3\\delta$")}


def fig1(S):
    ds = np.array(S["delta_R"])
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, v in zip(axs, S["v"]):
        ax.axhspan(0.99, 1.01, color="0.87", lw=0); ax.axhline(1, color="0.4", lw=0.7)
        for k in ["hard", "parabola", "tuned_rqf"]:
            c, m, ls, lab = STY[k]
            ax.plot(ds, S["ratio"][k][str(v)], color=c, marker=m, ms=2.5, lw=1.3, ls=ls, label=lab)
        ax.axvline(300, color="0.5", lw=0.7, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 2e2); ax.set_xlim(18, 3300)
        ax.set_title(f"$v$ = {v} mm/s", fontsize=9); ax.set_xlabel(r"cut-off position $\delta$ (nm)")
    axs[0].set_ylabel(r"$R\,/\,R_{\rm true}$"); axs[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig1_R_vs_delta.pdf")


def fig2(S):
    d = np.array(S["delta_amp"])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for k in ["hard", "parabola", "tuned_quintic", "default_rqf", "tuned_rqf"]:
        c, m, ls, lab = STY[k]
        e = np.array(S["amp"][k]["1.0"]); ok = (e < 0.5) & ((d <= 160) if k == "tuned_quintic" else True)
        ax.plot(d[ok], e[ok], color=c, marker=m, ms=3, lw=1.3, ls=ls, label=lab + f" ({S['orders'][k]:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500]))
    ax.xaxis.set_major_formatter(FixedFormatter(["20", "50", "100", "200", "500"])); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"cut-off position $\delta$ (nm)"); ax.set_ylabel(r"$|r_{\rm cut}-r_{\rm true}|$  ($v$ = 1 mm/s)")
    ax.legend(frameon=False, fontsize=7, loc="lower right", title="fitted order, 20–150 nm", title_fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig2_amplitude.pdf")


def fig3():
    sp = SPECIES["Rb"]; dd, w = 100.0, 100.0; Vd = abs(sp.V(dd))
    x = np.linspace(-40, 220, 1500)
    def prof(c):
        out = np.where(x >= dd, sp.V(np.maximum(x, dd)), c.F)
        a = (x >= c.xs) & (x < dd); out[a] = c.Vin(x[a]); return out / Vd
    Vh = prof(C.hard(sp, dd)); Vr = prof(C.rqf(sp, dd, w, 0.5)); Vn = prof(C.quintic(sp, dd, w, 0.1))
    xc = x[x > 60]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.plot(xc, sp.V(xc) / Vd, color=GREY, lw=1, label="CP potential")
    ax.plot(x, Vh, color=GREY, lw=1.5, ls="--", label="hard cut-off")
    ax.plot(x, Vn, color=RED, lw=1.8, label=r"non-monotone C$^2$ quintic ($u=0.1$)")
    ax.plot(x, Vr, color=BLUE, lw=1.8, label=r"monotone C$^2$ RQF ($u=0.5$)")
    i = np.argmin(Vn); ax.plot(x[i], Vn[i], "o", color=RED, ms=4)
    ax.annotate("spurious well; force\nrepulsive on its left flank", xy=(x[i], Vn[i]), xytext=(125, -3.0), fontsize=8,
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7))
    for xv, lab in [(0, r"$\delta-w$"), (100, r"$\delta$")]:
        ax.axvline(xv, color="0.6", lw=0.7, ls=":"); ax.text(xv + 3, -3.7, lab, fontsize=8)
    ax.set_ylim(-3.8, 1.1); ax.set_yticks([0, -1, -2, -3]); ax.set_xlim(-40, 220)
    ax.set_xlabel("distance $x$ (nm)"); ax.set_ylabel(r"$V/|V(\delta)|$")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "fig4_shapes.pdf")
    print("fig3 well depth below floor / |V(delta)|:", Vn[0] - Vn[i])


def fig4(Cst, sec_per_unit):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for k in ["hard", "parabola", "tuned_quintic", "default_rqf", "tuned_rqf"]:
        c, m, ls, lab = STY[k]
        rows = [r for r in Cst["fam"][k] if r["amp"] < 0.5 and (k != "tuned_quintic" or r["d"] <= 160)]
        hrs = np.array([r["cost"] for r in rows]) * sec_per_unit / 3600
        e = np.array([r["amp"] for r in rows])
        ax.plot(hrs, e, color=c, marker=m, ms=3, lw=1.3, ls=ls, label=lab)
        if k == "tuned_rqf":
            for r, h, ee in zip(rows, hrs, e):
                if round(r["d"]) in (20, 31, 49, 97, 191, 300):
                    ax.annotate(f"{r['d']:.0f} nm", (h, ee), textcoords="offset points", xytext=(-6, -11), fontsize=6.5, color=BLUE)
    for v, lab in [("1.0", "1% at 1 mm/s"), ("2.5", "1% at 2.5 mm/s")]:
        y = Cst["amp_1pct"][v]; ax.axhline(y, color="0.6", lw=0.7, ls="--")
        ax.text(3e3, y * 1.12, lab, fontsize=6.5, color="0.4", ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("estimated run time of a converged time-dependent run (CPU hours)")
    ax.set_ylabel(r"$|r_{\rm cut}-r_{\rm true}|$")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "fig3_cost.pdf")


if __name__ == "__main__":
    S = json.load(open(RES / "stationary.json"))
    fig1(S); fig2(S); fig3()
    if (RES / "cost.json").exists():
        Cst = json.load(open(RES / "cost.json"))
        fig4(Cst, Cst.get("sec_per_unit", 5e-9))
