"""
replay_and_compare.py
------------------------
This is the piece that turns "we can detect drift and score its severity"
into "reacting to it actually helps." Three things happen here:

1. A compressed memory buffer that keeps a small, class-balanced sample of
   old embeddings around (8-bit quantized, so it's genuinely tiny) - this
   is what lets the device "remember" old activities without storing raw
   sensor data.

2. Last-layer-only retraining - the Conv1D feature extractor stays frozen
   forever, only the final Dense(64->6) classifier head ever gets updated.
   That's the whole point of using embeddings in the first place: retraining
   a 6x64 layer is cheap, retraining the whole CNN would not be.

3. The actual three-way comparison your synopsis promises: fixed-schedule
   retraining vs retrain-on-every-flag vs your cost-gated approach (tau =
   0.15, picked in the last step). Same subject-pair streams as before, so
   the results plug directly into what you already have.

One assumption worth stating plainly (and putting in your report): this
simulation assumes true labels become available for a short recent window
whenever a retrain fires, since that's what "replay on new data" requires.
On the real deployed device this labeled window would need to come from
some other source (a calibration step, a user confirmation, etc.) - a
different, separate problem to be solved in Semester 2, not something this
script tries to answer.
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

SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

EMBED_DIM = 64
NUM_CLASSES = 6
TAU = 0.15                 # from threshold_selection.py
SEVERITY_WINDOW = 60       # widened from 30 - testing whether a wider window
                            # smooths over the bursty, per-activity error
                            # pattern that made pair 1->10 read as low-severity
                            # despite being the worst-performing subject overall
FIXED_INTERVAL = 100        # how often the "fixed schedule" baseline retrains
NEW_DATA_WINDOW = 30        # how many recent labeled samples feed a retrain
BUFFER_CAPACITY_PER_CLASS = 50   # 50 * 6 classes * 64 dims * 1 byte = ~19 KB total


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

model = keras.models.load_model(BASE_DIR / "cnn_baseline//har_cnn_full.keras")
mean = np.load(BASE_DIR / "cnn_baseline//norm_mean.npy")
std = np.load(BASE_DIR / "cnn_baseline//norm_std.npy")

embedding_extractor = keras.Model(model.input, model.get_layer("embedding").output)
W0, b0 = model.get_layer("classifier").get_weights()  # starting point every policy resets to

subject_ids = sorted(np.unique(subj_all))
all_embeddings = embedding_extractor.predict((X_all - mean) / std, verbose=0)
EMBED_MAX = all_embeddings.max()  # for quantization scale - ReLU output, so floor is 0 already


# ---------------------------------------------------------------------------
# The buffer. Reservoir sampling per class means: while there's room, just
# keep everything; once full, each new sample has a shrinking chance of
# bumping an old one out, so the buffer stays a fair, unbiased sample of
# everything it's ever seen instead of just "whatever came in last."
# ---------------------------------------------------------------------------
class LatentBuffer:
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
                # classic reservoir swap - replace a random existing slot with
                # probability capacity/seen, so older samples don't dominate
                # just because they got in first
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

    def size_bytes(self):
        return sum(self.filled[c] * EMBED_DIM for c in range(NUM_CLASSES))  # 1 byte/value, uint8


def get_ddm():
    if hasattr(drift, "DDM"):
        return drift.DDM()
    return drift.binary.DDM()


def retrain_head(W, b, replay_embeddings, replay_labels, epochs=15, lr=1e-3):
    """Trains ONLY the classifier Dense layer - starts from the current
    weights (not from scratch) so it's a fine-tune, not a fresh fit."""
    layer = keras.layers.Dense(NUM_CLASSES, activation="softmax")
    layer.build((None, EMBED_DIM))
    layer.set_weights([W, b])
    tiny_model = keras.Sequential([keras.Input((EMBED_DIM,)), layer])
    tiny_model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy")
    tiny_model.fit(replay_embeddings, replay_labels, epochs=epochs, batch_size=32, verbose=0)
    return layer.get_weights()


def run_policy(emb_stream, y_stream, drift_point, policy):
    W, b = W0.copy(), b0.copy()
    buffer = LatentBuffer(BUFFER_CAPACITY_PER_CLASS, EMBED_DIM, NUM_CLASSES, EMBED_MAX)
    buffer.add(emb_stream[:drift_point], y_stream[:drift_point])  # seed with "old" knowledge

    detector = get_ddm()
    errors = []
    retrain_events = 0

    for i in range(len(emb_stream)):
        logits = emb_stream[i] @ W + b
        pred = int(np.argmax(logits))
        err = int(pred != y_stream[i])
        errors.append(err)
        detector.update(err)

        should_retrain = False
        if policy == "fixed_schedule":
            should_retrain = (i + 1) % FIXED_INTERVAL == 0
        elif getattr(detector, "drift_detected", False):
            if policy == "every_flag":
                should_retrain = True
            elif policy == "cost_gate":
                baseline = np.mean(errors[:drift_point]) if drift_point > 0 else 0.0
                recent = np.mean(errors[max(0, i - SEVERITY_WINDOW + 1):i + 1])
                severity = float(np.clip((recent - baseline) / max(1e-6, 1 - baseline), 0, 1))
                should_retrain = severity >= TAU

        if should_retrain:
            new_start = max(0, i - NEW_DATA_WINDOW + 1)
            buffer.add(emb_stream[new_start:i + 1], y_stream[new_start:i + 1])
            replay_emb, replay_lab = buffer.get_all()
            W, b = retrain_head(W, b, replay_emb, replay_lab)
            retrain_events += 1

    errors = np.array(errors)
    return {
        "policy": policy,
        "retrain_events": retrain_events,
        "overall_accuracy": 1 - errors.mean(),
        "post_drift_accuracy": 1 - errors[drift_point:].mean(),
        "buffer_kb": round(buffer.size_bytes() / 1024, 2),
    }


# same "best A per distinct B" pairs used since Week 2, for consistency -
# recomputed here using the same starting classifier weights (W0, b0), since
# per-subject accuracy needs to reflect the untouched model, not any policy's
# retrained state
preds_all = (all_embeddings @ W0 + b0).argmax(axis=1)
per_subject_acc = {sid: np.mean(preds_all[subj_all == sid] == y_all[subj_all == sid]) for sid in subject_ids}

best_a_for_b = {}
for a, b_ in itertools.permutations(subject_ids, 2):
    d = per_subject_acc[a] - per_subject_acc[b_]
    if b_ not in best_a_for_b or d > best_a_for_b[b_][0]:
        best_a_for_b[b_] = (d, a)
ranked = sorted(((d, a, b_) for b_, (d, a) in best_a_for_b.items()), reverse=True)
top_pairs = [(a, b_) for _, a, b_ in ranked[:6]]

print(f"Running the three-way comparison across {len(top_pairs)} pairs "
      f"(this retrains the classifier head several times per pair, give it a minute)...\n")

results = []
for a, b_ in top_pairs:
    mask_a, mask_b = subj_all == a, subj_all == b_
    emb_stream = np.concatenate([all_embeddings[mask_a], all_embeddings[mask_b]])
    y_stream = np.concatenate([y_all[mask_a], y_all[mask_b]])
    drift_point = mask_a.sum()

    for policy in ("fixed_schedule", "every_flag", "cost_gate"):
        row = run_policy(emb_stream, y_stream, drift_point, policy)
        row.update({"subject_A": a, "subject_B": b_})
        results.append(row)
        print(f"  A={a} B={b_} [{policy:14s}] retrains={row['retrain_events']:2d}  "
              f"post_drift_acc={row['post_drift_accuracy']:.3f}  buffer={row['buffer_kb']} KB")

df = pd.DataFrame(results)
df.to_csv(BASE_DIR / f"week4_policy_comparison_window{SEVERITY_WINDOW}.csv", index=False)

print(f"\nSaved to: {BASE_DIR / f'week4_policy_comparison_window{SEVERITY_WINDOW}.csv'}\n")
print("--- Averaged across all pairs ---")
summary = df.groupby("policy").agg(
    avg_retrains=("retrain_events", "mean"),
    avg_post_drift_accuracy=("post_drift_accuracy", "mean"),
    avg_buffer_kb=("buffer_kb", "mean"),
).round(3)
print(summary)

print("\nHow to read this: cost_gate should show noticeably fewer retrains than")
print("every_flag, while keeping post_drift_accuracy close to it (not close to")
print("fixed_schedule or a no-retrain baseline). That gap - similar accuracy,")
print("fewer retrains - is the entire result your project is trying to prove.")