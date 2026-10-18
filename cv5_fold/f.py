"""
final_figures.py  -  3 clean figures for the presentation (replaces the 24 images)

Put this file in the cv5_fold folder (next to cv5_drift_per_stream.csv) and run:
    python cv5_fold/final_figures.py
Needs: pandas, numpy, matplotlib.   No retraining, it only reads the CSVs you already have.
Output: cv5_fold/final_figures/  ->  fig1_accuracy_change.png
                                      fig2_retrains.png
                                      fig3_tau_sweep.png

HOW THE GROUPS WORK
  Every stream (person A -> person B) is ranked by how well "no_adapt" does on it.
  The 4 groups are quarters of that ranking: Hardest = the quarter where no_adapt
  does worst (biggest drift), Easiest = the quarter where it does best.
  Whiskers = 95% bootstrap interval over streams (streams, not seeds, are the
  independent units).
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "final_figures")
os.makedirs(OUT, exist_ok=True)

# ---- look (light surface, validated categorical slots 1-3, neutral grey for the reference) ----
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6a66"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})

GATE = "cost_gate_old_tau0.15"          # your real gate (tau fixed in advance)
POLICIES = [                              # colour follows the policy in every figure
    (GATE, "Cost gate (tau 0.15)", BLUE),
    ("every_flag", "Retrain on every flag", ORANGE),
    ("fixed_schedule", "Fixed schedule", AQUA),
]
LABELS = ["Hardest", "Hard", "Easy", "Easiest"]
rng = np.random.default_rng(0)


def find(name):
    for base in (HERE, os.path.join(HERE, "cv5_fold"), os.getcwd(), os.path.join(os.getcwd(), "cv5_fold")):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {name}. Put this script next to it (in cv5_fold).")


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def boot_ci(x, n=4000):
    x = np.asarray(x, float)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


# ------------------------------------------------------------------ load stream-level data
d = pd.read_csv(find("cv5_drift_per_stream.csv"))
keys = list(d.columns[:list(d.columns).index("policy")])
stream_keys = [k for k in keys if k != "seed"]
acc = d.pivot_table(index=keys, columns="policy", values="post_drift_acc").groupby(level=stream_keys).mean()
ret = d.pivot_table(index=keys, columns="policy", values="retrains").groupby(level=stream_keys).mean()

rank = acc["no_adapt"].rank(method="first")
group = pd.qcut(rank, 4, labels=LABELS)
n_streams = len(acc)
print(f"{n_streams} streams, ~{n_streams // 4} per group (each averaged over seeds)\n")

# per group x policy: change in accuracy vs no_adapt (percentage points) with CI, and mean retrains
gain, retr = {}, {}
for g in LABELS:
    m = (group == g).to_numpy()
    for key, _, _ in POLICIES:
        diff = (acc[key] - acc["no_adapt"])[m].to_numpy() * 100
        lo, hi = boot_ci(diff)
        gain[(g, key)] = (diff.mean(), lo, hi)
        retr[(g, key)] = ret[key][m].mean()

print("Change in post-drift accuracy vs no_adapt (points) [95% CI]   |   retrains per stream")
for g in LABELS:
    cells = []
    for key, name, _ in POLICIES:
        mu, lo, hi = gain[(g, key)]
        cells.append(f"{name}: {mu:+.2f} [{lo:+.2f},{hi:+.2f}] / {retr[(g, key)]:.2f}")
    print(f"  {g:8s} " + "   ".join(cells))
print()


def grouped_bars(ax, values, cis=None, fmt="{:.1f}"):
    w = 0.24
    for j, (key, name, color) in enumerate(POLICIES):
        xs = np.arange(len(LABELS)) + (j - 1) * (w + 0.02)
        vals = np.array([values[(g, key)] for g in LABELS])
        ax.bar(xs, vals, width=w, color=color, edgecolor=SURFACE, linewidth=1.5, label=name, zorder=3)
        for i, v in enumerate(vals):
            top, bot = v, v
            if cis is not None:
                _, lo, hi = cis[(LABELS[i], key)]
                ax.errorbar(xs[i], v, yerr=[[v - lo], [hi - v]], color=INK2, lw=1, capsize=2.5, zorder=4)
                top, bot = hi, lo
            ax.annotate(("0.0" if abs(v) < 0.05 else fmt.format(v)), (xs[i], top if v >= 0 else bot),
                        xytext=(0, 4 if v >= 0 else -4), textcoords="offset points",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_xticklabels([f"{g}\n(quarter {i + 1} of 4)" for i, g in enumerate(LABELS)])
    ax.grid(axis="x", visible=False)


# ------------------------------------------------------------------ FIGURE 1
fig, ax = plt.subplots(figsize=(10.5, 5.8))
vals1 = {k: v[0] for k, v in gain.items()}
grouped_bars(ax, vals1, cis=gain, fmt="{:+.1f}")
ax.axhline(0, color=INK, lw=1.2, zorder=5)
lo_y, hi_y = ax.get_ylim()
ax.set_ylim(lo_y - 0.6, hi_y + 0.6)
ax.text(1.0, 0.0, "  no_adapt = 0", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=9, color=INK2)
ax.set_ylabel("Change in post-drift accuracy vs no_adapt\n(percentage points)")
ax.set_title("Retraining only pays off when the drift is big", loc="left", fontsize=14, fontweight="bold", pad=26)
ax.text(0, 1.03, "Streams grouped by how hard no_adapt finds them. Whiskers = 95% interval over streams.",
        transform=ax.transAxes, fontsize=9.5, color=INK2)
ax.legend(loc="upper right", frameon=False, fontsize=9.5)
save(fig, "fig1_accuracy_change.png")

# ------------------------------------------------------------------ FIGURE 2
fig, ax = plt.subplots(figsize=(10.5, 5.4))
grouped_bars(ax, retr, fmt="{:.1f}")
ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
ax.set_ylabel("Retrains per stream\n(lower = cheaper)")
ax.set_title("The gate spends its retrains where the drift is big", loc="left", fontsize=14, fontweight="bold", pad=26)
ax.text(0, 1.03, "Same groups as the previous chart. no_adapt never retrains (0).",
        transform=ax.transAxes, fontsize=9.5, color=INK2)
ax.legend(loc="upper right", frameon=False, fontsize=9.5)
save(fig, "fig2_retrains.png")

# ------------------------------------------------------------------ FIGURE 3
files = sorted(glob.glob(os.path.join(os.path.dirname(find("cv5_drift_per_stream.csv")), "cv5_tau_sweep_fold*.csv")))
sw = pd.concat([pd.read_csv(f) for f in files])
s = sw.groupby("tau")[["retrains", "post_acc", "base_acc", "every_acc", "every_retrains"]].mean().reset_index()
s[["post_acc", "base_acc", "every_acc"]] *= 100
tau, ref_tau = s["tau"].to_numpy(), 0.15

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
# left: cost
a1.plot(tau, s["retrains"], color=BLUE, lw=2, marker="o", ms=4, label="Cost gate", zorder=3)
a1.axhline(s["every_retrains"].mean(), color=ORANGE, lw=2, ls="--", label="Retrain on every flag")
a1.axhline(0, color=GREY, lw=2, ls=":", label="no_adapt (never retrains)")
a1.set_xlabel("tau (higher = pickier before retraining)")
a1.set_ylabel("Retrains per stream (lower = cheaper)")
a1.set_title("Cost: a higher tau means fewer retrains", loc="left", fontsize=12, fontweight="bold")
# right: accuracy
a2.plot(tau, s["post_acc"], color=BLUE, lw=2, marker="o", ms=4, label="Cost gate", zorder=3)
a2.axhline(s["every_acc"].mean(), color=ORANGE, lw=2, ls="--", label="Retrain on every flag")
a2.axhline(s["base_acc"].mean(), color=GREY, lw=2, ls=":", label="no_adapt")
a2.set_xlabel("tau (higher = pickier before retraining)")
a2.set_ylabel("Post-drift accuracy (%)  [axis zoomed]")
a2.set_title("Accuracy as tau changes", loc="left", fontsize=12, fontweight="bold")
lo, hi = s[["post_acc", "every_acc", "base_acc"]].min().min(), s[["post_acc", "every_acc", "base_acc"]].max().max()
a2.set_ylim(lo - 0.4, hi + 0.4)
# mark the tau you use
row = s.iloc[(s["tau"] - ref_tau).abs().argmin()]
for ax_, y in ((a1, row["retrains"]), (a2, row["post_acc"])):
    ax_.scatter([row["tau"]], [y], s=90, facecolor=SURFACE, edgecolor=BLUE, linewidth=2.2, zorder=5)
    ax_.annotate(f"tau {row['tau']:.2f}\n(your gate)", (row["tau"], y), xytext=(14, 14),
                 textcoords="offset points", fontsize=9, color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
h, l = a1.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=10)
fig.suptitle("Choosing tau: average of the 5 folds", x=0.01, y=1.10, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "fig3_tau_sweep.png")

print("\nDone. Open the 'final_figures' folder.")