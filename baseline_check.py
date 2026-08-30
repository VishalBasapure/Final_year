"""
severity_baseline_check.py
----------------------------
Every pair we tested so far used a subject A that the CNN gets 100% right,
so baseline_error_rate was always 0 in the results - we never actually
exercised the "subtract the baseline" half of the severity formula. This
script forces in some pairs where A is realistically imperfect (accuracy
in the low-to-mid 90s, not a perfect classifier) so we can see whether the
formula behaves sensibly once there's real baseline noise to account for.

Same severity formula as last time:
    severity = (recent_error_rate - baseline_error_rate) / (1 - baseline_error_rate)

If this is working the way it should: two pairs with roughly similar RAW
recent error rates but different baselines should NOT get the same
severity score - the one with the higher baseline should score lower,
because some of that error was already "normal" for that subject, not
new drift.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from river import drift

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "UCI HAR Dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
CNN_DIR = BASE_DIR / "cnn_baseline"

SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

SEVERITY_WINDOW = 30
# See severity_extraction.py for why this changed from a 100-sample tail to
# the full pre-drift segment - a small window let one unlucky/lucky stretch
# distort severity (this is the exact bug the 14->4 pair exposed).


def load_split(split_dir, split_name):
    channels = []
    for name in SIGNAL_NAMES:
        path = split_dir / "Inertial Signals" / f"{name}_{split_name}.txt"
        channels.append(pd.read_csv(path, sep=r"\s+", header=None).to_numpy())
    X = np.stack(channels, axis=-1)
    y = pd.read_csv(split_dir / f"y_{split_name}.txt", sep=r"\s+", header=None)[0].to_numpy()
    subjects = pd.read_csv(split_dir / f"subject_{split_name}.txt", sep=r"\s+", header=None)[0].to_numpy()
    return X, y, subjects


print("Loading data + trained CNN...")
X_tr, y_tr, subj_tr = load_split(TRAIN_DIR, "train")
X_te, y_te, subj_te = load_split(TEST_DIR, "test")

X_all = np.concatenate([X_tr, X_te])
y_all = np.concatenate([y_tr, y_te]) - 1
subj_all = np.concatenate([subj_tr, subj_te])

model = keras.models.load_model(CNN_DIR / "har_cnn_full.keras")
mean = np.load(CNN_DIR / "norm_mean.npy")
std = np.load(CNN_DIR / "norm_std.npy")
all_preds = model.predict((X_all - mean) / std, verbose=0).argmax(axis=1)

subject_ids = sorted(np.unique(subj_all))
per_subject_acc = {
    sid: np.mean(all_preds[subj_all == sid] == y_all[subj_all == sid])
    for sid in subject_ids
}

# these six A's are NOT perfect (roughly 0.91-0.96 accuracy from last week's
# printout) - paired against the same weak B's we already tested, so we can
# compare like-for-like against the earlier perfect-A results
FORCED_IMPERFECT_PAIRS = [
    (5, 10), (8, 16), (2, 9), (14, 4), (12, 25), (6, 7),
]

print("\nSubject A accuracies for the forced pairs (should all be < 1.0):")
for a, b in FORCED_IMPERFECT_PAIRS:
    print(f"  A={a}: {per_subject_acc[a]:.3f}   B={b}: {per_subject_acc[b]:.3f}")


def get_ddm():
    if hasattr(drift, "DDM"):
        return drift.DDM()
    return drift.binary.DDM()


def severity_at(errors, flag_idx, drift_point):
    baseline = np.mean(errors[:drift_point]) if drift_point > 0 else 0.0

    recent_start = max(0, flag_idx - SEVERITY_WINDOW + 1)
    recent = np.mean(errors[recent_start:flag_idx + 1])

    denom = max(1e-6, 1.0 - baseline)
    raw = (recent - baseline) / denom
    return float(np.clip(raw, 0.0, 1.0)), baseline, recent


def run_pair(subject_a, subject_b):
    mask_a = subj_all == subject_a
    mask_b = subj_all == subject_b

    preds_stream = np.concatenate([all_preds[mask_a], all_preds[mask_b]])
    y_stream = np.concatenate([y_all[mask_a], y_all[mask_b]])
    errors = (preds_stream != y_stream).astype(int)
    drift_point = mask_a.sum()

    detector = get_ddm()
    pair_rows = []
    for i, err in enumerate(errors):
        detector.update(err)
        if getattr(detector, "drift_detected", False):
            sev, baseline_err, recent_err = severity_at(errors, i, drift_point)
            pair_rows.append({
                "subject_A": subject_a, "subject_B": subject_b,
                "flag_index": i, "samples_after_true_drift": i - drift_point,
                "baseline_error_rate": round(baseline_err, 3),
                "recent_error_rate": round(recent_err, 3),
                "severity": round(sev, 3),
            })
    return pair_rows


all_rows = []
for a, b in FORCED_IMPERFECT_PAIRS:
    all_rows.extend(run_pair(a, b))

results = pd.DataFrame(all_rows)
out_path = BASE_DIR / "week3_severity_imperfect_baseline.csv"
results.to_csv(out_path, index=False)

print(f"\nSaved to: {out_path}\n")
print(results.to_string(index=False))

if len(results):
    zero_baseline_avg_sev = results.loc[results.baseline_error_rate == 0, "severity"].mean()
    nonzero_baseline_avg_sev = results.loc[results.baseline_error_rate > 0, "severity"].mean()
    print("\nQuick check:")
    print(f"  Avg severity where baseline_error_rate == 0: {zero_baseline_avg_sev:.3f}")
    print(f"  Avg severity where baseline_error_rate  > 0: {nonzero_baseline_avg_sev:.3f}")
    print("  If the second number isn't systematically lower for similar recent_error_rate,")
    print("  look at individual rows - the baseline term should be pulling severity down")
    print("  whenever some of the recent error was already 'normal' for that subject.")
else:
    print("\nNo flags fired for any of these pairs - try different A/B combinations.")
