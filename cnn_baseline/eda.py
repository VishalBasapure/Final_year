"""
eda.py - Exploratory Data Analysis for the UCI HAR dataset (raw inertial signals).

Run from the F_project folder:
    python eda.py

Outputs go to ./eda_outputs/ (PNG plots, channel_stats.csv, eda_summary.txt).
Needs: numpy, pandas, matplotlib. Section 7 also uses scikit-learn (skipped if missing).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "eda_outputs"
os.makedirs(OUT, exist_ok=True)
FS = 50  # sampling rate in Hz (from the dataset README)
CH = ["body_acc_x", "body_acc_y", "body_acc_z",
      "body_gyro_x", "body_gyro_y", "body_gyro_z",
      "total_acc_x", "total_acc_y", "total_acc_z"]
ACT = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

LOG = []


def say(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    LOG.append(text)


def box(ax, data, labels):
    """Boxplot that works on any matplotlib version (avoids labels/tick_labels rename)."""
    ax.boxplot(data)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)


def header(title):
    say("")
    say("=" * 70)
    say(title)
    say("=" * 70)


def find_root():
    """Find the folder that holds activity_labels.txt (skips .venv and __MACOSX junk)."""
    for d, dirs, files in os.walk("."):
        dirs[:] = [x for x in dirs if x not in (".venv", "__MACOSX", ".git", "__pycache__")]
        if "activity_labels.txt" in files and "train" in dirs and "test" in dirs:
            return d
    raise FileNotFoundError("Could not find the UCI HAR Dataset folder. Run from F_project.")


def read_table(path):
    return pd.read_csv(path, sep=r"\s+", header=None).values


def load_split(root, split):
    sig = np.stack(
        [read_table(os.path.join(root, split, "Inertial Signals", f"{c}_{split}.txt")) for c in CH],
        axis=-1)  # (windows, 128 timesteps, 9 channels)
    y = read_table(os.path.join(root, split, f"y_{split}.txt")).ravel().astype(int)
    s = read_table(os.path.join(root, split, f"subject_{split}.txt")).ravel().astype(int)
    return sig, y, s


# ---------------------------------------------------------------- 1. SHAPE
root = find_root()
header("1. SHAPE OF THE DATA")
say("Dataset folder:", os.path.abspath(root))
Xtr, ytr, str_ = load_split(root, "train")
Xte, yte, ste = load_split(root, "test")
say(f"Train signals: {Xtr.shape}  (windows, timesteps, channels)")
say(f"Test  signals: {Xte.shape}")
say(f"Window length: {Xtr.shape[1]} samples = {Xtr.shape[1] / FS:.2f} s at {FS} Hz")
say(f"Channels ({len(CH)}): {CH}")
say(f"Train subjects ({len(set(str_))}): {sorted(set(str_))}")
say(f"Test  subjects ({len(set(ste))}): {sorted(set(ste))}")
say("Train/test subjects overlap?", bool(set(str_) & set(ste)), "(False = split is by subject)")
say("Classes:", dict(zip(range(1, 7), ACT)))
say("Task type: 6-class CLASSIFICATION")

try:
    f561 = read_table(os.path.join(root, "train", "X_train.txt"))
    say(f"Precomputed feature file X_train.txt: {f561.shape} (561 hand-made features per window)")
    say("  NaNs in X_train.txt:", int(np.isnan(f561).sum()))
    del f561
except Exception as e:
    say("Could not read X_train.txt:", e)

X = np.concatenate([Xtr, Xte])
y = np.concatenate([ytr, yte])
subj = np.concatenate([str_, ste])
is_train = np.concatenate([np.ones(len(ytr), bool), np.zeros(len(yte), bool)])

# ------------------------------------------------- 2. QUALITY + PER-COLUMN STATS
header("2. DATA QUALITY AND PER-CHANNEL STATISTICS")
say("NaNs:", int(np.isnan(X).sum()), "| Infs:", int(np.isinf(X).sum()))
flat = X.reshape(-1, X.shape[-1])
stats = pd.DataFrame({
    "min": flat.min(0), "max": flat.max(0), "mean": flat.mean(0), "std": flat.std(0),
    "skew": pd.DataFrame(flat).skew().values,
}, index=CH)
stats.to_csv(os.path.join(OUT, "channel_stats.csv"))
say(stats.round(4).to_string())
say("Units: acc channels are in g (total_acc includes gravity, body_acc has gravity removed), gyro in rad/s.")

a, b = X[1:, :64, 6], X[:-1, 64:, 6]
overlap = np.all(np.isclose(a, b), axis=1) & (subj[1:] == subj[:-1])
say(f"Consecutive windows sharing the same 64 samples (50% overlap): {overlap.mean() * 100:.1f}% of pairs")
say("  -> overlapping windows mean RANDOM k-fold would leak data. Use subject-wise folds.")

# ------------------------------------------------------ 3. CLASS BALANCE
header("3. CLASS BALANCE AND SUBJECTS")
cnt = pd.DataFrame({"train": pd.Series(ytr).value_counts().sort_index(),
                    "test": pd.Series(yte).value_counts().sort_index()})
cnt.index = ACT
cnt["total"] = cnt.sum(axis=1)
cnt["pct"] = (cnt["total"] / cnt["total"].sum() * 100).round(1)
say(cnt.to_string())
say(f"Imbalance ratio (largest/smallest class): {cnt['total'].max() / cnt['total'].min():.2f}")
per_subj = pd.Series(subj).value_counts().sort_index()
say(f"Windows per subject: min={per_subj.min()}, max={per_subj.max()}, mean={per_subj.mean():.0f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
cnt[["train", "test"]].plot.bar(ax=ax[0])
ax[0].set_title("Windows per activity")
ax[0].set_ylabel("count")
tab = pd.crosstab(subj, y).reindex(columns=range(1, 7), fill_value=0)
im = ax[1].imshow(tab.values, aspect="auto", cmap="viridis")
ax[1].set_yticks(range(len(tab.index)))
ax[1].set_yticklabels(tab.index, fontsize=7)
ax[1].set_xticks(range(6))
ax[1].set_xticklabels(ACT, rotation=45, ha="right")
ax[1].set_title("Windows per subject x activity")
plt.colorbar(im, ax=ax[1])
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_class_balance.png"), dpi=130)
plt.close()

# ------------------------------------------------------------ 4. DYNAMICS
header("4. DYNAMICS (how signals behave over time)")
rng = np.random.default_rng(42)
t = np.arange(X.shape[1]) / FS
groups = [("total_acc", [6, 7, 8]), ("body_acc", [0, 1, 2]), ("body_gyro", [3, 4, 5])]
fig, axes = plt.subplots(6, 3, figsize=(15, 14), sharex=True)
for r, k in enumerate(range(1, 7)):
    idx = rng.choice(np.where(y == k)[0])
    for c, (name, chs) in enumerate(groups):
        for ch, lab in zip(chs, "xyz"):
            axes[r, c].plot(t, X[idx, :, ch], label=lab, lw=1)
        if r == 0:
            axes[r, c].set_title(name)
        if c == 0:
            axes[r, c].set_ylabel(ACT[k - 1], fontsize=9)
axes[0, 0].legend(fontsize=7)
axes[-1, 0].set_xlabel("time (s)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_example_windows.png"), dpi=120)
plt.close()

mag = np.sqrt((X[:, :, 0:3] ** 2).sum(-1))             # body_acc magnitude per timestep
energy = mag.std(axis=1)                                 # how much movement in the window
spec = np.abs(np.fft.rfft(mag - mag.mean(1, keepdims=True), axis=1))
freqs = np.fft.rfftfreq(X.shape[1], 1 / FS)
domf = freqs[spec[:, 1:].argmax(1) + 1]                  # dominant frequency, DC skipped
dyn = pd.DataFrame({"activity": [ACT[k - 1] for k in y], "energy": energy, "dom_freq_hz": domf})
say(dyn.groupby("activity")[["energy", "dom_freq_hz"]].agg(["mean", "std"]).round(3).loc[ACT].to_string())

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
box(ax[0], [energy[y == k] for k in range(1, 7)], ACT)
ax[0].set_title("Movement energy (std of body-acc magnitude)")
ax[0].set_yscale("log")
box(ax[1], [domf[y == k] for k in range(1, 7)], ACT)
ax[1].set_title("Dominant frequency (Hz)")
for a_ in ax:
    a_.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_energy_frequency.png"), dpi=130)
plt.close()

# ------------------------------------------------------- 5. RELATIONSHIPS
header("5. RELATIONSHIPS BETWEEN COLUMNS (channels)")
corr = np.corrcoef(flat.T)
cdf = pd.DataFrame(corr, index=CH, columns=CH)
say(cdf.round(2).to_string())
pairs = [(CH[i], CH[j], corr[i, j]) for i in range(9) for j in range(i + 1, 9)]
pairs.sort(key=lambda p: -abs(p[2]))
say("Strongest correlated channel pairs:")
for p in pairs[:5]:
    say(f"  {p[0]} <-> {p[1]}: {p[2]:.2f}")

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(9))
ax.set_xticklabels(CH, rotation=60, ha="right")
ax.set_yticks(range(9))
ax.set_yticklabels(CH)
for i in range(9):
    for j in range(9):
        ax.text(j, i, f"{corr[i, j]:.1f}", ha="center", va="center", fontsize=7)
plt.colorbar(im)
ax.set_title("Channel correlation")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_channel_correlation.png"), dpi=130)
plt.close()

# ----------------------------------------------- 6. SUBJECT VARIABILITY (drift)
header("6. SUBJECT-TO-SUBJECT VARIABILITY (why drift exists in your project)")
subs = sorted(set(subj))
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
ax = ax.ravel()
box(ax[0], [energy[(y == 1) & (subj == s)] for s in subs], subs)
ax[0].set_title("WALKING: movement energy per subject")
for n, (ch, lab) in enumerate(zip([6, 7, 8], "xyz"), start=1):
    wm = X[:, :, ch].mean(1)
    box(ax[n], [wm[(y == 6) & (subj == s)] for s in subs], subs)
    ax[n].set_title(f"LAYING: mean total_acc_{lab} per subject")
for a_ in ax:
    a_.tick_params(axis="x", labelsize=6)
    a_.set_xlabel("subject id")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_subject_variability.png"), dpi=130)
plt.close()
sub_walk = pd.Series({s: energy[(y == 1) & (subj == s)].mean() for s in subs})
say("Walking energy, mean per subject: min", round(sub_walk.min(), 3), "max", round(sub_walk.max(), 3),
    f"(ratio {sub_walk.max() / sub_walk.min():.2f}x)")
say("Bigger spread between subjects = same activity looks different = drift when the user changes.")

# ------------------------------------------------------ 7. FEATURE PROMINENCE
header("7. WHICH FEATURES MATTER? (quick check, not your CNN)")
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix

    F = np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1)  # 36 summary features
    names = [f"{s}_{c}" for s in ("mean", "std", "min", "max") for c in CH]
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(F[is_train], y[is_train])
    pred = rf.predict(F[~is_train])
    say(f"Random forest on 36 summary features, test accuracy: {accuracy_score(y[~is_train], pred):.3f}")
    say("Confusion matrix (rows=true, cols=pred):")
    say(pd.DataFrame(confusion_matrix(y[~is_train], pred), index=ACT, columns=ACT).to_string())
    imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
    say("Top 10 features:")
    say(imp.head(10).round(4).to_string())
    chan_imp = {c: sum(imp[f"{s}_{c}"] for s in ("mean", "std", "min", "max")) for c in CH}
    chan_imp = pd.Series(chan_imp).sort_values(ascending=False)
    say("Importance summed per channel:")
    say(chan_imp.round(4).to_string())
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    imp.head(15)[::-1].plot.barh(ax=ax[0])
    ax[0].set_title("Top 15 summary features")
    chan_imp[::-1].plot.barh(ax=ax[1])
    ax[1].set_title("Importance per channel")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "06_feature_importance.png"), dpi=130)
    plt.close()
except ImportError:
    say("scikit-learn not installed, skipping. Install with: pip install scikit-learn")

with open(os.path.join(OUT, "eda_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
print(f"\nDone. Everything saved in ./{OUT}/")