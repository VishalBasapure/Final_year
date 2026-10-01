"""
cost_benefit_selection.py
----------------------------
threshold_selection.py picked tau using severity alone - "how bad does this
look" - completely blind to "would retraining actually help." That's the
missing half of the plan's cost-benefit equation:

    benefit_score = severity x accuracy_gain_from_retrain

(divided by energy cost, once we have a real ESP32-S3 + INA219 measurement -
for now energy cost is held at a constant 1, so this script isolates the
severity x accuracy_gain half first, same way threshold_selection.py isolated
severity alone.)

For every DDM flag (real detection AND false alarm - we need both, so false
alarms with a low accuracy_gain still get correctly discouraged), this asks:
"if we retrain right here, how much better does accuracy get on the next
LOOKAHEAD_WINDOW samples, compared to NOT retraining?"

Simplifying assumption (stating this plainly, same as replay_compare.py's
docstring does for its own assumption): each flag is evaluated as if it were
the FIRST retrain decision in that pair's stream - buffer seeded only with
the pre-drift segment, weights starting from W0/b0 (untouched). This keeps
every flag's evaluation independent and directly comparable, rather than
compounding with whatever a specific policy would have already done earlier
in the stream. Reusing the buffer/retrain machinery from replay_compare.py
means results here plug in directly against what's already been measured.

v2 CHANGES (week5b) - two fixes after the first run made the benefit-aware
gate WORSE than severity-only (3 of 17 kept vs 6 of 17):

  1. LEAK FIX. The old version seeded the buffer with emb_stream[:drift_point]
     for every flag. For a false alarm (flag BEFORE the drift point) that
     buffer already contained the lookahead window it was then tested on -
     training on the exam questions. The buffer is now seeded only with
     data the device would actually have seen by the flag (everything
     before the recent window, capped at drift_point), then the recent
     window is added.

  2. CONTROL (PLACEBO) GAIN. Retraining on recent same-subject data can
     raise accuracy even when nothing drifted (plain fine-tuning benefit).
     So for each pair we also measure the gain from the SAME retrain
     procedure at several points inside the pre-drift segment, where there
     is no drift by construction, and average them into control_gain.
         accuracy_gain = raw_gain - control_gain
     i.e. only the part of the gain that fine-tuning alone does NOT explain
     counts as "retraining fixed drift". Set USE_CONTROL = False to switch
     this off and see the effect of the leak fix on its own (good ablation
     for the report).
"""

import copy
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from river import drift

SEED = 42
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "UCI HAR Dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

EMBED_DIM = 64
NUM_CLASSES = 6
SEVERITY_WINDOW = 30        # locked-in value from replay_compare.py
NEW_DATA_WINDOW = 30        # same window used to feed a real retrain
LOOKAHEAD_WINDOW = 100       # how far past the flag we measure "did retraining help"
BUFFER_CAPACITY_PER_CLASS = 50
TOP_N_PAIRS = 6
USE_CONTROL = False          # False = leak fix only (ablation)
N_CONTROL_POINTS = 5        # placebo retrains per pair, spread over pre-drift data


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

model = keras.models.load_model(BASE_DIR / "cnn_baseline" / "har_cnn_full.keras")
mean = np.load(BASE_DIR / "cnn_baseline" / "norm_mean.npy")
std = np.load(BASE_DIR / "cnn_baseline" / "norm_std.npy")

embedding_extractor = keras.Model(model.input, model.get_layer("embedding").output)
W0, b0 = model.get_layer("classifier").get_weights()

subject_ids = sorted(np.unique(subj_all))
all_embeddings = embedding_extractor.predict((X_all - mean) / std, verbose=0)
EMBED_MAX = all_embeddings.max()


class LatentBuffer:
    """Identical to replay_compare.py's buffer - same 8-bit quantization,
    same reservoir sampling - so results here are directly comparable."""

    def __init__(self, capacity_per_class, embed_dim, num_classes, quant_max):
        self.capacity = capacity_per_class
        self.quant_max = quant_max
        self.slots = {c: np.zeros((capacity_per_class, embed_dim), dtype=np.uint8) for c in range(num_classes)}
        self.filled = {c: 0 for c in range(num_classes)}
        self.seen = {c: 0 for c in range(num_classes)}

    def _quantize(self, embeddings):
        scaled = np.clip(embeddings / self.quant_max, 0, 1) * 255
        return scaled.astype(np.uint8)

    def _dequantize(self, q):
        return (q.astype(np.float32) / 255.0) * self.quant_max

    def add(self, embeddings, labels):
        q = self._quantize(embeddings)
        for emb, label in zip(q, labels):
            c = int(label)
            self.seen[c] += 1
            if self.filled[c] < self.capacity:
                self.slots[c][self.filled[c]] = emb
                self.filled[c] += 1
            else:
                j = np.random.randint(0, self.seen[c])
                if j < self.capacity:
                    self.slots[c][j] = emb

    def get_all(self):
        embs, labs = [], []
        for c in range(NUM_CLASSES):
            n = self.filled[c]
            if n:
                embs.append(self._dequantize(self.slots[c][:n]))
                labs.append(np.full(n, c))
        return np.concatenate(embs), np.concatenate(labs)


def get_ddm():
    if hasattr(drift, "DDM"):
        return drift.DDM()
    return drift.binary.DDM()


def retrain_head(W, b, replay_embeddings, replay_labels, epochs=15, lr=1e-3):
    layer = keras.layers.Dense(NUM_CLASSES, activation="softmax")
    layer.build((None, EMBED_DIM))
    layer.set_weights([W, b])
    tiny_model = keras.Sequential([keras.Input((EMBED_DIM,)), layer])
    tiny_model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy")
    tiny_model.fit(replay_embeddings, replay_labels, epochs=epochs, batch_size=32,
                   shuffle=False, verbose=0)
    return layer.get_weights()


def accuracy_on(W, b, emb, y):
    if len(y) == 0:
        return None
    preds = (emb @ W + b).argmax(axis=1)
    return float(np.mean(preds == y))


def severity_at(errors, flag_idx, drift_point):
    baseline = np.mean(errors[:drift_point]) if drift_point > 0 else 0.0
    recent_start = max(0, flag_idx - SEVERITY_WINDOW + 1)
    recent = np.mean(errors[recent_start:flag_idx + 1])
    denom = max(1e-6, 1.0 - baseline)
    raw = (recent - baseline) / denom
    return float(np.clip(raw, 0.0, 1.0))


def gain_at(emb_stream, y_stream, flag_idx, seed_cap):
    """Accuracy gain (retrain vs do-nothing) on the LOOKAHEAD_WINDOW after
    flag_idx. Buffer = only data the device has already seen: everything
    before the recent window (capped at seed_cap), then the recent window.
    Returns None if there is no lookahead window left."""
    end = min(len(emb_stream), flag_idx + 1 + LOOKAHEAD_WINDOW)
    lookahead_emb = emb_stream[flag_idx + 1:end]
    lookahead_y = y_stream[flag_idx + 1:end]
    if len(lookahead_y) == 0:
        return None

    acc_without = accuracy_on(W0, b0, lookahead_emb, lookahead_y)

    new_start = max(0, flag_idx - NEW_DATA_WINDOW + 1)
    seed_end = min(seed_cap, new_start)
    buffer = LatentBuffer(BUFFER_CAPACITY_PER_CLASS, EMBED_DIM, NUM_CLASSES, EMBED_MAX)
    if seed_end > 0:
        buffer.add(emb_stream[:seed_end], y_stream[:seed_end])
    buffer.add(emb_stream[new_start:flag_idx + 1], y_stream[new_start:flag_idx + 1])
    replay_emb, replay_lab = buffer.get_all()
    W_new, b_new = retrain_head(W0.copy(), b0.copy(), replay_emb, replay_lab)
    acc_with = accuracy_on(W_new, b_new, lookahead_emb, lookahead_y)
    return acc_with - acc_without


def control_gain_for_pair(emb_stream, y_stream, drift_point):
    """Placebo: same retrain procedure at points inside the pre-drift
    segment, with the lookahead also fully pre-drift. Any gain here is
    generic fine-tuning benefit, NOT drift correction."""
    if not USE_CONTROL:
        return 0.0, []
    lo = 2 * NEW_DATA_WINDOW
    hi = drift_point - LOOKAHEAD_WINDOW - 1
    if hi < lo:
        return 0.0, []
    points = np.unique(np.linspace(lo, hi, N_CONTROL_POINTS).astype(int))
    gains = []
    for c in points:
        g = gain_at(emb_stream, y_stream, int(c), seed_cap=int(c) + 1)
        if g is not None:
            gains.append(g)
    return (float(np.mean(gains)) if gains else 0.0), gains


preds_all = (all_embeddings @ W0 + b0).argmax(axis=1)
per_subject_acc = {sid: np.mean(preds_all[subj_all == sid] == y_all[subj_all == sid]) for sid in subject_ids}

best_a_for_b = {}
for a, b_ in itertools.permutations(subject_ids, 2):
    d = per_subject_acc[a] - per_subject_acc[b_]
    if b_ not in best_a_for_b or d > best_a_for_b[b_][0]:
        best_a_for_b[b_] = (d, a)
ranked = sorted(((d, a, b_) for b_, (d, a) in best_a_for_b.items()), reverse=True)
top_pairs = [(a, b_) for _, a, b_ in ranked[:TOP_N_PAIRS]]

# same forced imperfect-baseline pairs as baseline_check.py - without these,
# every auto-selected pair above has a perfect subject A (acc=1.000), so
# NONE of them ever produce a false alarm. That made the first run of this
# script look artificially clean ("avoids all 0 false alarms" - trivially
# true when there are zero false alarms to avoid). Adding these back in is
# what makes this comparable to threshold_selection.py's combined dataset.
FORCED_IMPERFECT_PAIRS = [
    (5, 10), (8, 16), (2, 9), (14, 4), (12, 25), (6, 7),
]
top_pairs = top_pairs + FORCED_IMPERFECT_PAIRS

print(f"Evaluating every DDM flag across {len(top_pairs)} pairs "
      f"(retrains once per flag to measure accuracy_gain, give it a few minutes)...\n")

rows = []
for subject_a, subject_b in top_pairs:
    mask_a, mask_b = subj_all == subject_a, subj_all == subject_b
    emb_stream = np.concatenate([all_embeddings[mask_a], all_embeddings[mask_b]])
    y_stream = np.concatenate([y_all[mask_a], y_all[mask_b]])
    preds_stream = (emb_stream @ W0 + b0).argmax(axis=1)
    errors = (preds_stream != y_stream).astype(int)
    drift_point = mask_a.sum()

    control_gain, control_list = control_gain_for_pair(emb_stream, y_stream, drift_point)
    print(f"  A={subject_a} B={subject_b} control_gain={control_gain:+.3f} "
          f"(from {len(control_list)} placebo retrains)")

    detector = get_ddm()
    for i, err in enumerate(errors):
        detector.update(err)
        if getattr(detector, "drift_detected", False):
            sev = severity_at(errors, i, drift_point)
            raw_gain = gain_at(emb_stream, y_stream, i, seed_cap=drift_point)
            if raw_gain is None:
                continue  # flag too near the end of the stream, skip it
            gain = raw_gain - control_gain
            rows.append({
                "subject_A": subject_a,
                "subject_B": subject_b,
                "flag_index": i,
                "samples_after_true_drift": i - drift_point,
                "severity": round(sev, 3),
                "raw_gain": round(raw_gain, 3),
                "control_gain": round(control_gain, 3),
                "accuracy_gain": round(gain, 3),
                "benefit_score": round(sev * gain, 4),
            })
    print(f"  done: A={subject_a} B={subject_b}")
    print(y_stream[330:400])
data = pd.DataFrame(rows)
data["is_false_alarm"] = data["samples_after_true_drift"] < 0
data["is_real_detection"] = ~data["is_false_alarm"]
out_path = BASE_DIR / "week5b_cost_benefit_values.csv"
data.to_csv(out_path, index=False)

print(f"\nSaved to: {out_path}\n")
print(data.to_string(index=False))

# same sweep logic as threshold_selection.py, but on benefit_score instead
# of severity alone
candidate_taus = np.round(np.arange(data.benefit_score.min(), data.benefit_score.max() + 0.01, 0.01), 3)
sweep_rows = []
for tau in candidate_taus:
    gated_in = data["benefit_score"] >= tau
    sweep_rows.append({
        "tau": tau,
        "false_alarms_avoided": int((data.is_false_alarm & ~gated_in).sum()),
        "false_alarms_still_retrained_on": int((data.is_false_alarm & gated_in).sum()),
        "real_detections_kept": int((data.is_real_detection & gated_in).sum()),
        "real_detections_ignored": int((data.is_real_detection & ~gated_in).sum()),
    })
sweep = pd.DataFrame(sweep_rows)
sweep.to_csv(BASE_DIR / "week5b_benefit_tau_sweep.csv", index=False)

clean = sweep[sweep.false_alarms_still_retrained_on == 0]
print("\n--- Benefit-score tau sweep (severity x accuracy_gain) ---")
print(sweep.to_string(index=False))
print(y_stream[330:400])
if len(clean):
    best = clean.sort_values("real_detections_kept", ascending=False).iloc[0]
    print(f"\nRecommended benefit-score tau = {best.tau}")
    print(f"  -> avoids all {int(data.is_false_alarm.sum())} false alarms")
    print(f"  -> keeps {int(best.real_detections_kept)} of {int(data.is_real_detection.sum())} real detections")
    print("\nCompare this real_detections_kept number against threshold_selection.py's")
    print("severity-only result (6 of 17 kept at tau=0.15). If this number is higher,")
    print("that's the concrete proof the benefit-aware gate is less trigger-shy than")
    print("the severity-only one - put both numbers side by side in your report.")
else:
    print("\nNo tau fully avoids every false alarm on benefit_score either - inspect")
    print("week5b_cost_benefit_values.csv and pick the best manual trade-off.")

print("\nNext: plug this new tau back into replay_compare.py in place of the old")
print("severity>=0.15 rule (severity*accuracy_gain >= new_tau instead), rerun the")
print("three-way policy comparison, and see whether cost_gate's retrain count goes")
print("up from ~0.2-0.3 while post_drift_accuracy gets closer to every_flag's. That")
print("comparison is your evidence that the fuller cost-benefit gate is better than")
print("the severity-only one - exactly what Niteesh asked you to formalize.")