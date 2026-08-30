"""
Week 2 - Drift Detector Comparison (CNN-powered, auto-loops over subject pairs)
=================================================================================
Goal: stop hand-picking subject pairs. This script:
  1. Loads your trained CNN + normalization stats from Week 1.
  2. Measures the CNN's accuracy on EVERY subject individually.
  3. Auto-ranks subject pairs by how much accuracy drops between them -
     i.e. finds the strongest "new user" drift scenarios by itself.
  4. Builds a synthetic stream (subject A's data, then subject B's data) for
     the top N pairs, runs ADWIN / DDM / Page-Hinkley on each, and records
     detection delay + false-positive count for every (pair, detector)
     combination.
  5. Saves everything to a CSV - this table IS your Week 2 deliverable.

Concept note: since the CNN is one global model (not retrained per subject
like the RandomForest smoke test was), "drift" here means "the shared model
suddenly starts seeing a harder user" - which is exactly the real-world
scenario your project targets (a wearable passed to a new person).

Requires: har_cnn_full.keras, norm_mean.npy, norm_std.npy already saved by
cnn_baseline/cnn_baseline.py, sitting in the same folder as this script.
"""

import itertools
import os
import numpy as np
import pandas as pd
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow import keras
from river import drift

tf.get_logger().setLevel("ERROR")


def get_ddm():
    # river moved binary-input detectors (DDM, EDDM, HDDM_A/W) into
    # river.drift.binary at some point, so the top-level path breaks on
    # newer installs. Try both so this doesn't rot the next time someone
    # updates the package.
    if hasattr(drift, "DDM"):
        return drift.DDM()
    return drift.binary.DDM()

# ---------------------------------------------------------------------------
# 0. Paths + config
# ---------------------------------------------------------------------------
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

TOP_N_PAIRS = 6          # how many strongest-drift pairs to fully test
DETECTION_WINDOW = 300   # a flag counts as "detecting" the drift if it fires
                          # within this many samples after the true drift point


# ---------------------------------------------------------------------------
# 1. Load raw signals (same loader as cnn_baseline.py) and POOL train+test
#    together, so we have all 30 subjects available to pick pairs from.
# ---------------------------------------------------------------------------
def load_split(split_dir, split_name):
    channels = []
    for name in SIGNAL_NAMES:
        path = split_dir / "Inertial Signals" / f"{name}_{split_name}.txt"
        arr = pd.read_csv(path, sep=r"\s+", header=None).to_numpy()
        channels.append(arr)
    X = np.stack(channels, axis=-1)  # (N, 128, 9)
    y = pd.read_csv(split_dir / f"y_{split_name}.txt", sep=r"\s+", header=None)[0].to_numpy()
    subjects = pd.read_csv(split_dir / f"subject_{split_name}.txt", sep=r"\s+", header=None)[0].to_numpy()
    return X, y, subjects


print("Loading data + trained model...")
X_tr, y_tr, subj_tr = load_split(TRAIN_DIR, "train")
X_te, y_te, subj_te = load_split(TEST_DIR, "test")

X_all = np.concatenate([X_tr, X_te], axis=0)
y_all = np.concatenate([y_tr, y_te], axis=0) - 1  # back to 0-5 for the model
subj_all = np.concatenate([subj_tr, subj_te], axis=0)

model = keras.models.load_model(CNN_DIR / "har_cnn_full.keras", compile=False)
mean = np.load(CNN_DIR / "norm_mean.npy")
std = np.load(CNN_DIR / "norm_std.npy")

X_all_norm = (X_all - mean) / std
all_preds = model.predict(X_all_norm, verbose=0).argmax(axis=1)

print(f"Loaded {X_all.shape[0]} total windows across {len(np.unique(subj_all))} subjects")


# ---------------------------------------------------------------------------
# 2. Per-subject accuracy - this replaces manually eyeballing a sweep table
# ---------------------------------------------------------------------------
subject_ids = sorted(np.unique(subj_all))
per_subject_acc = {}
for sid in subject_ids:
    mask = subj_all == sid
    per_subject_acc[sid] = np.mean(all_preds[mask] == y_all[mask])

print("\nPer-subject accuracy (global CNN):")
for sid, acc in sorted(per_subject_acc.items(), key=lambda kv: kv[1]):
    print(f"  Subject {sid:>2}: {acc:.3f}")


# ---------------------------------------------------------------------------
# 3. Auto-rank pairs by accuracy drop (A = well-served subject, B = harder
#    subject) - this is the auto version of last week's manual sweep.
# ---------------------------------------------------------------------------
# For each candidate "hard" subject B, keep only its single best-matched A
# (the A with the biggest accuracy gap against that B). Without this, a
# handful of near-perfect A subjects paired with one unusually hard B would
# flood the top of the list and you'd end up testing the same B over and
# over instead of getting a spread of different drift scenarios.
best_a_for_b = {}
for a, b in itertools.permutations(subject_ids, 2):
    drop = per_subject_acc[a] - per_subject_acc[b]
    if b not in best_a_for_b or drop > best_a_for_b[b][0]:
        best_a_for_b[b] = (drop, a)

ranked_by_b = sorted(
    ((drop, a, b) for b, (drop, a) in best_a_for_b.items()),
    reverse=True,
)
top_pairs = [(a, b) for _, a, b in ranked_by_b[:TOP_N_PAIRS]]

print(f"\nTop {TOP_N_PAIRS} auto-selected strong-drift pairs:")
for a, b in top_pairs:
    print(f"  A={a} (acc={per_subject_acc[a]:.3f}) -> B={b} (acc={per_subject_acc[b]:.3f}), "
          f"drop={per_subject_acc[a] - per_subject_acc[b]:.3f}")


# ---------------------------------------------------------------------------
# 4. Detector setup + helper functions
# ---------------------------------------------------------------------------
def make_detectors():
    # fresh instances every call - these things carry internal state across
    # updates, so reusing one object between pairs would bleed history from
    # pair N into pair N+1 and quietly wreck the comparison
    return {
        "ADWIN": drift.ADWIN(),
        "DDM": get_ddm(),
        "PageHinkley": drift.PageHinkley(threshold=10, delta=0.001),
    }


def run_detector(detector, errors):
    flags = []
    for i, err in enumerate(errors):
        detector.update(err)
        if getattr(detector, "drift_detected", False):
            flags.append(i)
    return flags


def evaluate_pair(subject_a, subject_b):
    mask_a = subj_all == subject_a
    mask_b = subj_all == subject_b

    preds_a, y_a = all_preds[mask_a], y_all[mask_a]
    preds_b, y_b = all_preds[mask_b], y_all[mask_b]

    stream_preds = np.concatenate([preds_a, preds_b])
    stream_y = np.concatenate([y_a, y_b])
    errors = (stream_preds != stream_y).astype(int)
    drift_point = len(preds_a)

    err_before = errors[:drift_point].mean()
    err_after = errors[drift_point:].mean()

    rows = []
    for name, detector in make_detectors().items():
        flags = run_detector(detector, errors)

        false_positives = [f for f in flags if f < drift_point]
        true_detections = [f for f in flags if drift_point <= f < drift_point + DETECTION_WINDOW]

        if true_detections:
            delay = true_detections[0] - drift_point
        else:
            delay = None  # missed the drift within the detection window

        rows.append({
            "subject_A": subject_a, "subject_B": subject_b,
            "detector": name,
            "acc_A": per_subject_acc[subject_a], "acc_B": per_subject_acc[subject_b],
            "error_before": round(err_before, 3), "error_after": round(err_after, 3),
            "total_flags": len(flags),
            "false_positives_pre_drift": len(false_positives),
            "detected_within_window": delay is not None,
            "detection_delay_samples": delay,
        })
    return rows


# ---------------------------------------------------------------------------
# 5. Run the comparison across all top pairs
# ---------------------------------------------------------------------------
print(f"\nRunning ADWIN / DDM / PageHinkley across {TOP_N_PAIRS} pairs...")
all_rows = []
for a, b in top_pairs:
    all_rows.extend(evaluate_pair(a, b))

results = pd.DataFrame(all_rows)
csv_path = BASE_DIR / "week2_drift_detector_comparison.csv"
results.to_csv(csv_path, index=False)

print(f"\nSaved full results table to: {csv_path}")
print("\n--- Per-detector summary (averaged across all pairs) ---")
summary = results.groupby("detector").agg(
    pairs_detected=("detected_within_window", "sum"),
    total_pairs=("detected_within_window", "count"),
    avg_delay=("detection_delay_samples", "mean"),
    avg_false_positives=("false_positives_pre_drift", "mean"),
).round(2)
print(summary)

print("\nHow to read this:")
print("- pairs_detected/total_pairs: how reliably each detector caught real drift")
print("- avg_delay: lower is better (faster reaction to real drift)")
print("- avg_false_positives: lower is better (fewer wasted alarms before drift even happened)")
print("\nThis table is your Week 2 deliverable - pick a 'winner' detector based on")
print("the best balance of reliability, low delay, and low false positives, and")
print("state why in your report. That detector feeds into Week 3's cost-gate.")
