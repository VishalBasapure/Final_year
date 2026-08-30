"""
severity_extraction.py
-----------------------
Last week we picked DDM as our drift detector - fastest, zero false
positives, across six genuinely different subject pairs. Good. But DDM
only tells you YES/NO, a change happened. Your project needs more than
that: it needs to know HOW SERIOUS the change was, because that's the
whole point of the cost-gate - react hard to a serious drift, shrug off
a minor one.

I looked at pulling DDM's internal p/s statistics directly (that's what
the original paper's math is built on), but river keeps those as private
attributes that aren't guaranteed stable across versions - and given we've
already been burned twice this week by river restructuring itself, I'm
not going to hang your core contribution on undocumented internals.

So instead: severity gets computed independently, straight from the error
stream, with a formula you can explain to your guide in one sentence -
"how far did the recent error rate climb above the subject's normal
baseline, as a fraction of the room it had left to climb."

    severity = (recent_error_rate - baseline_error_rate) / (1 - baseline_error_rate)

Sits between 0 (no worse than normal) and 1 (every single recent prediction
is wrong). This is the number Week 3's cost-gate will actually threshold
against - this script just gets it on the table so we can look at real
values before picking that threshold.
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

TOP_N_PAIRS = 6        # same diverse-pair selection as last week's comparison
SEVERITY_WINDOW = 30   # how many samples back we look to judge "recent" error rate
# Baseline used to be just the last 100 pre-drift samples, but that's a small
# enough slice that a subject can get a lucky or unlucky stretch just by
# chance - which then throws off severity for anything measured against it.
# Since subject A's pre-drift segment has no drift inside it by construction
# (same person the whole time), using the FULL segment as baseline gives a
# far more stable "what's normal" estimate, with no downside - there's no
# recency argument for preferring a short window when the whole thing is
# equally valid.


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

# same "best A per hard B" pair selection from the detector comparison,
# so the pairs here line up with last week's results and we're not
# suddenly comparing against a different set of scenarios
best_a_for_b = {}
for a, b in itertools.permutations(subject_ids, 2):
    drop = per_subject_acc[a] - per_subject_acc[b]
    if b not in best_a_for_b or drop > best_a_for_b[b][0]:
        best_a_for_b[b] = (drop, a)

ranked = sorted(((d, a, b) for b, (d, a) in best_a_for_b.items()), reverse=True)
top_pairs = [(a, b) for _, a, b in ranked[:TOP_N_PAIRS]]


def get_ddm():
    # same defensive fallback as last week - don't assume where DDM lives
    if hasattr(drift, "DDM"):
        return drift.DDM()
    return drift.binary.DDM()


def severity_at(errors, flag_idx, drift_point):
    """The actual severity formula. Baseline now comes from the WHOLE
    pre-drift segment, not a short tail - see the note above on why a
    small window turned out to be a real problem, not a style choice."""
    baseline = np.mean(errors[:drift_point]) if drift_point > 0 else 0.0

    recent_start = max(0, flag_idx - SEVERITY_WINDOW + 1)
    recent = np.mean(errors[recent_start:flag_idx + 1])

    denom = max(1e-6, 1.0 - baseline)  # guard against baseline == 1.0 (shouldn't happen but still)
    raw = (recent - baseline) / denom
    return float(np.clip(raw, 0.0, 1.0)), baseline, recent


rows = []
for subject_a, subject_b in top_pairs:
    mask_a = subj_all == subject_a
    mask_b = subj_all == subject_b

    preds_stream = np.concatenate([all_preds[mask_a], all_preds[mask_b]])
    y_stream = np.concatenate([y_all[mask_a], y_all[mask_b]])
    errors = (preds_stream != y_stream).astype(int)
    drift_point = mask_a.sum()

    detector = get_ddm()
    for i, err in enumerate(errors):
        detector.update(err)
        if getattr(detector, "drift_detected", False):
            sev, baseline_err, recent_err = severity_at(errors, i, drift_point)
            rows.append({
                "subject_A": subject_a,
                "subject_B": subject_b,
                "flag_index": i,
                "samples_after_true_drift": i - drift_point,
                "baseline_error_rate": round(baseline_err, 3),
                "recent_error_rate": round(recent_err, 3),
                "severity": round(sev, 3),
            })

results = pd.DataFrame(rows)
out_path = BASE_DIR / "week3_severity_values.csv"
results.to_csv(out_path, index=False)

print(f"\nCollected {len(results)} drift flags across {TOP_N_PAIRS} pairs.")
print(f"Saved to: {out_path}\n")
print(results.to_string(index=False))

print("\nWhat to look at:")
print("- Do genuinely severe drift pairs (big accuracy drop) end up with high severity scores?")
print("  If not, the formula needs adjusting before we pick a threshold.")
print("- The spread of severity values here is what we'll use next to pick tau -")
print("  the cutoff above which the cost-gate says 'this is worth retraining for'.")
