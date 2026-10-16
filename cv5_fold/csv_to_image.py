"""
csv_to_images.py
Run from any folder:   python cv5_fold/csv_to_image.py
Reads every cv5_*.csv and writes PNG images into  cv_images/
(chart where it makes sense + a clean table image for every file).
Needs: pandas, matplotlib, numpy  (pip install pandas matplotlib numpy)
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "cv_images")
os.makedirs(OUT, exist_ok=True)
DPI = 200


def save(fig, name):
    path = os.path.join(OUT, name + ".png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


def find_col(df, *keys, exclude=()):
    """first column whose name contains any key (case-insensitive)."""
    for c in df.columns:
        low = str(c).lower()
        if any(k in low for k in keys) and not any(e in low for e in exclude):
            return c
    return None


def table_image(df, name, title, max_rows=30):
    d = df.copy()
    shown = d.head(max_rows)
    for c in shown.select_dtypes("number").columns:
        if pd.api.types.is_float_dtype(shown[c]):
            shown[c] = shown[c].round(3)
    w = max(6, 1.3 * len(shown.columns))
    h = max(2, 0.4 * (len(shown) + 2))
    fig, ax = plt.subplots(figsize=(w, h))
    ax.axis("off")
    tbl = ax.table(cellText=shown.astype(str).values,
                   colLabels=[str(c) for c in shown.columns],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for (r, _), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2b4c7e")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eef2f7")
    note = f"  (first {max_rows} of {len(d)} rows)" if len(d) > max_rows else ""
    ax.set_title(title + note, fontsize=11, weight="bold", pad=10)
    save(fig, name + "_table")


# ---------- specific charts ----------
def drift_per_stream(df, name):
    pol = find_col(df, "policy")
    acc = find_col(df, "post_drift_acc", "acc")
    ret = find_col(df, "retrain", exclude=("before",))
    fgt = find_col(df, "forget")
    if pol is None or acc is None:
        return
    order = list(dict.fromkeys(df[pol]))
    metrics = [(m, lbl) for m, lbl in
               [(acc, "Post-drift accuracy"), (ret, "Retrains per stream"),
                (fgt, "Forgetting")] if m is not None]

    # 1) policy summary bars (mean +- std over all streams)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (m, lbl) in zip(axes, metrics):
        g = df.groupby(pol)[m].agg(["mean", "std"]).reindex(order)
        ax.bar(range(len(g)), g["mean"], yerr=g["std"].fillna(0), capsize=3,
               color=plt.cm.tab10(np.arange(len(g))))
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
        ax.set_title(lbl, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Policy comparison (mean ± std over all streams)", weight="bold")
    save(fig, name + "_policy_summary")

    # 2) per-fold grouped bars of accuracy
    fold = find_col(df, "fold")
    if fold is not None:
        g = df.groupby([fold, pol])[acc].mean().unstack(pol)[order]
        fig, ax = plt.subplots(figsize=(10, 4.8))
        g.plot(kind="bar", ax=ax, width=0.85)
        ax.set_ylabel("Post-drift accuracy")
        ax.set_xlabel("Fold")
        ax.set_title("Post-drift accuracy per fold, by policy", weight="bold")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=0)
        save(fig, name + "_per_fold")

    # 3) summary table (the numbers behind chart 1)
    agg = {m: ["mean", "std"] for m, _ in metrics}
    s = df.groupby(pol).agg(agg).reindex(order)
    s.columns = [f"{a} ({b})" for a, b in s.columns]
    table_image(s.reset_index(), name + "_policy_summary",
                "Policy summary", max_rows=50)

    # 4) retrains vs accuracy trade-off
    if ret is not None:
        g = df.groupby(pol)[[ret, acc]].mean().reindex(order)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for i, (p, r) in enumerate(g.iterrows()):
            ax.scatter(r[ret], r[acc], s=90, color=plt.cm.tab10(i))
            ax.annotate(p, (r[ret], r[acc]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Avg retrains per stream (lower = cheaper)")
        ax.set_ylabel("Avg post-drift accuracy (higher = better)")
        ax.set_title("Cost vs accuracy trade-off", weight="bold")
        ax.grid(alpha=0.3)
        save(fig, name + "_tradeoff")


def confusion(df, name):
    d = df.copy()
    if not pd.api.types.is_numeric_dtype(d.iloc[:, 0]):
        d = d.set_index(d.columns[0])
    d = d.select_dtypes("number")
    if d.empty:
        return
    m = d.values
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(m.shape[1]))
    ax.set_yticks(range(m.shape[0]))
    ax.set_xticklabels(d.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(d.index, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:g}", ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black",
                    fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (5-fold CV)", weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, name + "_heatmap")


def bar_by_first_col(df, name, title):
    """e.g. cnn_folds (fold -> accuracy) or per_subject (subject -> accuracy)."""
    key = df.columns[0]
    nums = [c for c in df.select_dtypes("number").columns if c != key]
    pref = [c for c in nums if "acc" in str(c).lower()] or nums[:1]
    if not pref:
        return
    c = pref[0]
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(df)), 4.5))
    ax.bar(df[key].astype(str), df[c], color="#2b4c7e")
    ax.axhline(df[c].mean(), color="red", ls="--", lw=1,
               label=f"mean = {df[c].mean():.3f}")
    ax.set_xlabel(str(key))
    ax.set_ylabel(str(c))
    ax.set_title(title, weight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, name + "_bar")


def tau_sweep(df, name):
    tau = find_col(df, "tau", "thresh")
    if tau is None:
        return
    nums = [c for c in df.select_dtypes("number").columns if c != tau]
    if not nums:
        return
    n = len(nums)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, nums):
        ax.plot(df[tau], df[c], marker="o", ms=3)
        ax.set_xlabel(str(tau))
        ax.set_ylabel(str(c))
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Tau sweep: {name}", weight="bold")
    fig.tight_layout()
    save(fig, name + "_curves")


def main():
    files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "cv5_*.csv")))
    if not files:
        print(f"No cv5_*.csv files found in {SCRIPT_DIR}.")
        return
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        print(f"\n{f}: {df.shape[0]} rows x {df.shape[1]} cols")
        if "confusion" in name:
            confusion(df, name)
        elif "drift_per_stream" in name:
            drift_per_stream(df, name)
        elif "tau_sweep" in name:
            tau_sweep(df, name)
        elif "cnn_folds" in name:
            bar_by_first_col(df, name, "CNN accuracy per fold")
        elif "per_subject" in name:
            bar_by_first_col(df, name, "CNN accuracy per subject")
        # every file also gets a readable table image
        if "drift_per_stream" not in name:
            table_image(df, name, name)
    print(f"\nDone. Open the '{OUT}' folder.")


if __name__ == "__main__":
    main()