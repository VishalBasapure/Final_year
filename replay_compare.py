"""
replay_and_compare.py   (v2 - week 6: tau sweep + imperfect-baseline pairs)
---------------------------------------------------------------------------
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

3. The policy comparison your synopsis promises. Policies compared:
       never_retrain   - reference: the untouched CNN head, no adaptation
       fixed_schedule  - retrain every FIXED_INTERVAL samples, drift or not
       every_flag      - retrain on every DDM flag
       cost_gate_tauX  - retrain only if DDM flags AND severity >= X,
                         for each X in TAU_VALUES (a sweep, not one guess)

WHAT CHANGED FROM v1 (and why)
  * TAU SWEEP. v1 only tested tau=0.15, which retrained 0 times in most
    pairs (so it behaved like "never retrain"). Sweeping 0.15 / 0.10 / 0.05
    shows the whole retrains-vs-accuracy trade-off curve instead of one
    point on it.
  * IMPERFECT-BASELINE PAIRS ADDED. v1 used only the 6 auto-selected pairs,
    all with a perfect-accuracy subject A (acc = 1.000). Those pairs never
    produce false alarms, so a wasted retrain could never show up. The 6
    forced imperfect pairs (same list as baseline_check.py and
    cost_benefit.py) are added so false alarms exist and cost something.
  * WASTED-RETRAIN METRIC. Any retrain that fires BEFORE the true drift
    point is counted as wasted (nothing had drifted yet).
  * never_retrain REFERENCE POLICY added so "accuracy gained" has a floor.
  * PER-RUN RESEEDING. Every (pair, policy) run now starts from the same
    random state, so a result no longer depends on which runs came before
    it. Side effect: numbers for the original 6 pairs can differ slightly
    from the week4 CSV (the reservoir-sampling random stream is different).
    The week4 files are left untouched; this script writes week6_* files.

Assumptions worth stating plainly (and putting in your report):
  * Labels: this simulation assumes true labels become available for a
    short recent window whenever a retrain fires, since that's what "replay
    on new data" requires. On the real device this labeled window would
    need to come from some other source (a calibration step, a user
    confirmation, etc.) - a Semester 2 problem, not solved here.
  * Severity baseline: for flags after the drift point, the baseline error
    rate is computed from the pre-drift segment, which uses the true
    drift_point. A real device would not know drift_point; it would need a
    rolling baseline instead. Same simplification as the severity scripts.
  * Stream order: the UCI HAR windows inside a subject come in long
    same-activity blocks, so the "stream" is not a realistic mixed-activity
    timeline. Results are directional evidence, not deployment numbers.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from river import drift

# same reasoning as cnn_baseline.py - this script has real randomness (the
# buffer's reservoir swaps). Seeded once here, and AGAIN at the start of
# every run_policy() call so each run is reproducible on its own.
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
TAU_VALUES = [0.15, 0.10, 0.05]   # 0.15 = week 3 pick; 0.10 / 0.05 = less trigger-shy
SEVERITY_WINDOW = 30       # locked in (window=60 was tested and dropped)
FIXED_INTERVAL = 100        # how often the "fixed schedule" baseline retrains
NEW_DATA_WINDOW = 30        # how many recent labeled samples feed a retrain
BUFFER_CAPACITY_PER_CLASS = 50   # 50 * 6 classes * 64 dims * 1 byte = ~19 KB total
TOP_N_PAIRS = 6

# same forced imperfect-baseline pairs as baseline_check.py / cost_benefit.py
FORCED_IMPERFECT_PAIRS = [
    (5, 10), (8, 16), (2, 9), (14, 4), (12, 25), (6, 7),
]

# (policy, tau) - tau only matters for cost_gate
POLICY_SPECS = (
    [("never_retrain", None), ("fixed_schedule", None), ("every_flag", None)]
    + [("cost_gate", t) for t in TAU_VALUES]
)


def policy_label(policy, tau):
    return f"cost_gate_tau{tau:.2f}" if policy == "cost_gate" else policy


POLICY_ORDER = [policy_label(p, t) for p, t in POLICY_SPECS]


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
    tiny_model.fit(replay_embeddings, replay_labels, epochs=epochs, batch_size=32,
                   shuffle=False, verbose=0)  # shuffle=False so this step is reproducible too
    return layer.get_weights()


def run_policy(emb_stream, y_stream, drift_point, policy, tau=None):
    # fresh random state per run -> result doesn't depend on run order
    np.random.seed(SEED)

    W, b = W0.copy(), b0.copy()
    buffer = LatentBuffer(BUFFER_CAPACITY_PER_CLASS, EMBED_DIM, NUM_CLASSES, EMBED_MAX)
    buffer.add(emb_stream[:drift_point], y_stream[:drift_point])  # seed with "old" knowledge

    detector = get_ddm()
    errors = []
    retrain_events = 0
    wasted_retrains = 0   # retrains that fired BEFORE the true drift point

    for i in range(len(emb_stream)):
        logits = emb_stream[i] @ W + b
        pred = int(np.argmax(logits))
        err = int(pred != y_stream[i])
        errors.append(err)
        detector.update(err)

        should_retrain = False
        if policy == "never_retrain":
            should_retrain = False
        elif policy == "fixed_schedule":
            should_retrain = (i + 1) % FIXED_INTERVAL == 0
        elif getattr(detector, "drift_detected", False):
            if policy == "every_flag":
                should_retrain = True
            elif policy == "cost_gate":
                baseline = np.mean(errors[:drift_point]) if drift_point > 0 else 0.0
                recent = np.mean(errors[max(0, i - SEVERITY_WINDOW + 1):i + 1])
                severity = float(np.clip((recent - baseline) / max(1e-6, 1 - baseline), 0, 1))
                should_retrain = severity >= tau

        if should_retrain:
            new_start = max(0, i - NEW_DATA_WINDOW + 1)
            buffer.add(emb_stream[new_start:i + 1], y_stream[new_start:i + 1])
            replay_emb, replay_lab = buffer.get_all()
            W, b = retrain_head(W, b, replay_emb, replay_lab)
            retrain_events += 1
            if i < drift_point:
                wasted_retrains += 1

    errors = np.array(errors)
    return {
        "policy": policy_label(policy, tau),
        "tau": tau,
        "retrain_events": retrain_events,
        "wasted_retrains": wasted_retrains,
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
auto_pairs = [(a, b_) for _, a, b_ in ranked[:TOP_N_PAIRS]]

pairs = [(a, b_, "auto") for a, b_ in auto_pairs] + \
        [(a, b_, "forced") for a, b_ in FORCED_IMPERFECT_PAIRS]

print(f"Running {len(POLICY_SPECS)} policies across {len(pairs)} pairs "
      f"({len(auto_pairs)} auto + {len(FORCED_IMPERFECT_PAIRS)} forced imperfect-baseline). "
      f"Retrains the classifier head many times, give it a few minutes...\n")

results = []
for a, b_, pair_type in pairs:
    mask_a, mask_b = subj_all == a, subj_all == b_
    emb_stream = np.concatenate([all_embeddings[mask_a], all_embeddings[mask_b]])
    y_stream = np.concatenate([y_all[mask_a], y_all[mask_b]])
    drift_point = int(mask_a.sum())

    for policy, tau in POLICY_SPECS:
        row = run_policy(emb_stream, y_stream, drift_point, policy, tau)
        row.update({"subject_A": a, "subject_B": b_, "pair_type": pair_type})
        results.append(row)
        print(f"  A={a:2d} B={b_:2d} [{row['policy']:18s}] retrains={row['retrain_events']:2d} "
              f"(wasted={row['wasted_retrains']})  post_drift_acc={row['post_drift_accuracy']:.3f}  "
              f"buffer={row['buffer_kb']} KB")

df = pd.DataFrame(results)
per_pair_path = BASE_DIR / "week6_policy_tau_sweep_per_pair.csv"
df.to_csv(per_pair_path, index=False)


def summarize(frame):
    s = frame.groupby("policy").agg(
        avg_retrains=("retrain_events", "mean"),
        avg_wasted_retrains=("wasted_retrains", "mean"),
        avg_post_drift_accuracy=("post_drift_accuracy", "mean"),
        avg_buffer_kb=("buffer_kb", "mean"),
    ).reindex(POLICY_ORDER)
    s["acc_vs_never_retrain"] = (
        s["avg_post_drift_accuracy"] - s.loc["never_retrain", "avg_post_drift_accuracy"]
    )
    return s.round(3)


pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

summary_all = summarize(df)
summary_auto = summarize(df[df.pair_type == "auto"])
summary_forced = summarize(df[df.pair_type == "forced"])

summary_path = BASE_DIR / "week6_policy_tau_sweep_summary.csv"
pd.concat({"all_pairs": summary_all, "auto_pairs": summary_auto,
           "forced_pairs": summary_forced}).to_csv(summary_path)

print(f"\nSaved per-pair results to: {per_pair_path}")
print(f"Saved summaries to:        {summary_path}")

print(f"\n--- ALL {len(pairs)} pairs ---")
print(summary_all)
print(f"\n--- AUTO pairs only (A=1 etc., perfect-accuracy subject A -> no false alarms seen) ---")
print(summary_auto)
print(f"\n--- FORCED imperfect-baseline pairs only (false alarms exist here) ---")
print(summary_forced)

# retrains-vs-accuracy trade-off picture (optional - skipped if matplotlib missing)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, row in summary_all.iterrows():
        ax.scatter(row.avg_retrains, row.avg_post_drift_accuracy, s=70, zorder=3)
        ax.annotate(label, (row.avg_retrains, row.avg_post_drift_accuracy),
                    textcoords="offset points", xytext=(6, 5), fontsize=8)
    gate_rows = summary_all.loc[[l for l in POLICY_ORDER if l.startswith("cost_gate")]]
    ax.plot(gate_rows.avg_retrains, gate_rows.avg_post_drift_accuracy,
            linestyle="--", alpha=0.5, zorder=2)
    ax.set_xlabel("Average retrains per stream (lower = cheaper)")
    ax.set_ylabel("Average post-drift accuracy (higher = better)")
    ax.set_title(f"Retrains vs accuracy across {len(pairs)} subject pairs")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = BASE_DIR / "week6_policy_tradeoff.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nSaved trade-off plot to: {plot_path}")
except ImportError:
    print("\n(matplotlib not installed - skipped the trade-off plot)")

print("\nHow to read this:")
print("- never_retrain is the floor: accuracy you get for free. acc_vs_never_retrain")
print("  is how much each policy adds on top of that.")
print("- Good cost-gate = accuracy close to every_flag / fixed_schedule with far")
print("  fewer retrains. Look for the tau where the curve stops paying off.")
print("- avg_wasted_retrains = retrains that fired before any drift existed.")
print("  fixed_schedule wastes some by design (it ignores drift). For the gated")
print("  policies it only rises where false alarms exist (the FORCED pairs), so a")
print("  tau that lets it creep up is too low.")
print("- The AUTO-only table is the comparable-to-week4 view; the ALL-pairs table")
print("  is the honest one because it includes pairs where false alarms exist.")