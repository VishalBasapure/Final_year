"""
replay_multiseed.py   (week 7: held-out pairs + multi-seed statistics)
----------------------------------------------------------------------
Same simulation as replay_compare.py (buffer, last-layer retraining, DDM,
six policies) - the week6 result stays reproducible from that file, which is
left untouched. This script adds the two things that make the result
credible instead of just suggestive:

1. HELD-OUT PAIRS. tau=0.10 was chosen by looking at the 12 pairs used in
   weeks 3-6, then judged on those same 12 - optimistic. Here 6 NEW pairs,
   with subjects that appear nowhere in the 12, are run with the tau already
   fixed (FINAL_TAU). The pairs are written down below, before any results,
   with a simple rule: 3 pairs with a perfect-accuracy subject A (like the
   "auto" pairs) and 3 with an imperfect subject A (like the "forced"
   pairs), each B being an unused subject with below-average accuracy.
   The script refuses to run if any held-out subject overlaps the 12.

2. MULTIPLE SEEDS. Each (pair, policy) is run once per seed in SEEDS. The
   randomness being varied: which samples the reservoir buffer keeps, and
   the batch order inside each retrain (RETRAIN_SHUFFLE). The CNN itself is
   fixed. Reported: the mean, the spread ACROSS SEEDS (is the result stable
   run to run?) and the spread ACROSS PAIRS (does it depend on which
   subjects?).

PAIRED COMPARISON. For each pair, accuracy (averaged over seeds) of a gate
policy is subtracted from that of a baseline, giving one difference per
pair. Reported: mean difference, a 95% bootstrap interval over pairs,
and in how many pairs the gate is better / tied / worse (tie = within
TIE_TOL accuracy). If the interval vs every_flag contains 0, the honest
reading is "accuracy not distinguishable from every_flag"; the saving then
comes from the retrain ratio.

Reproducing week6: set SEEDS = [42], RETRAIN_SHUFFLE = False and look at the
"tuned_12" rows - they should match week6_policy_tau_sweep_summary.csv.
With RETRAIN_SHUFFLE = True (default here) numbers shift slightly because
the batch order inside each retrain now varies with the seed.

Assumptions carried over from replay_compare.py (state them in the report):
labels are available for a short recent window at retrain time; severity
baseline uses the true drift_point; the stream is activity blocks, not a
realistic mixed timeline. Everything is a directional simulation, not a
deployment measurement.
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
TAU_VALUES = [0.15, 0.10, 0.05]
FINAL_TAU = 0.10            # chosen in week 6 on the 12 tuning pairs - NOT re-picked here
SEVERITY_WINDOW = 30
FIXED_INTERVAL = 100
NEW_DATA_WINDOW = 30
BUFFER_CAPACITY_PER_CLASS = 50
TOP_N_PAIRS = 6

SEEDS = [42, 7, 2024]
RETRAIN_SHUFFLE = True      # False = deterministic retrains (seeds then only vary the reservoir)
N_BOOTSTRAP = 10000
TIE_TOL = 0.005             # accuracy differences within this count as a tie

FORCED_IMPERFECT_PAIRS = [
    (5, 10), (8, 16), (2, 9), (14, 4), (12, 25), (6, 7),
]

# Written down BEFORE seeing any result. A = 11, 15, 19 have accuracy 1.000;
# A = 13, 30, 17 are imperfect (0.960-0.965); every B is an unused subject
# with below-average accuracy (0.901-0.958). No subject overlaps the 12
# tuning pairs (checked at run time).
HELDOUT_PAIRS = [(11, 28), (15, 29), (19, 23), (13, 21), (30, 18), (17, 20)]

POLICY_SPECS = (
    [("never_retrain", None), ("fixed_schedule", None), ("every_flag", None)]
    + [("cost_gate", t) for t in TAU_VALUES]
)


def policy_label(policy, tau):
    return f"cost_gate_tau{tau:.2f}" if policy == "cost_gate" else policy


def short_label(policy, tau):
    return f"g{tau:.2f}" if policy == "cost_gate" else {"never_retrain": "never",
                                                       "fixed_schedule": "fixed",
                                                       "every_flag": "every"}[policy]


POLICY_ORDER = [policy_label(p, t) for p, t in POLICY_SPECS]
FINAL_LABEL = policy_label("cost_gate", FINAL_TAU)


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
    """Identical to replay_compare.py - 8-bit quantized, reservoir sampled."""

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

    def size_bytes(self):
        return sum(self.filled[c] * EMBED_DIM for c in range(NUM_CLASSES))


def get_ddm():
    if hasattr(drift, "DDM"):
        return drift.DDM()
    return drift.binary.DDM()


def retrain_head(W, b, replay_embeddings, replay_labels, epochs=15, lr=1e-3):
    """Trains ONLY the classifier Dense layer, fine-tuning from current weights."""
    layer = keras.layers.Dense(NUM_CLASSES, activation="softmax")
    layer.build((None, EMBED_DIM))
    layer.set_weights([W, b])
    tiny_model = keras.Sequential([keras.Input((EMBED_DIM,)), layer])
    tiny_model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy")
    tiny_model.fit(replay_embeddings, replay_labels, epochs=epochs, batch_size=32,
                   shuffle=RETRAIN_SHUFFLE, verbose=0)
    return layer.get_weights()


def run_policy(emb_stream, y_stream, drift_point, policy, tau, seed):
    # every (pair, policy, seed) run starts from its own fixed random state
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

    W, b = W0.copy(), b0.copy()
    buffer = LatentBuffer(BUFFER_CAPACITY_PER_CLASS, EMBED_DIM, NUM_CLASSES, EMBED_MAX)
    buffer.add(emb_stream[:drift_point], y_stream[:drift_point])

    detector = get_ddm()
    errors = []
    retrain_events = 0
    wasted_retrains = 0

    for i in range(len(emb_stream)):
        pred = int(np.argmax(emb_stream[i] @ W + b))
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


# ---- pair lists: same 12 tuning pairs as week 6, plus the held-out pairs ----
preds_all = (all_embeddings @ W0 + b0).argmax(axis=1)
per_subject_acc = {sid: np.mean(preds_all[subj_all == sid] == y_all[subj_all == sid]) for sid in subject_ids}

best_a_for_b = {}
for a, b_ in itertools.permutations(subject_ids, 2):
    d = per_subject_acc[a] - per_subject_acc[b_]
    if b_ not in best_a_for_b or d > best_a_for_b[b_][0]:
        best_a_for_b[b_] = (d, a)
ranked = sorted(((d, a, b_) for b_, (d, a) in best_a_for_b.items()), reverse=True)
auto_pairs = [(a, b_) for _, a, b_ in ranked[:TOP_N_PAIRS]]

tuning_subjects = {s for pair in auto_pairs + FORCED_IMPERFECT_PAIRS for s in pair}
heldout_subjects = {s for pair in HELDOUT_PAIRS for s in pair}
overlap = tuning_subjects & heldout_subjects
if overlap:
    raise ValueError(f"Held-out pairs reuse tuning subjects {sorted(overlap)} - "
                     f"pick different subjects so the held-out test is fair.")

print("Held-out pairs (subject accuracy of the untouched CNN):")
for a, b_ in HELDOUT_PAIRS:
    print(f"  A={a:2d} (acc {per_subject_acc[a]:.3f})  ->  B={b_:2d} (acc {per_subject_acc[b_]:.3f})")

pairs = ([(a, b_, "auto") for a, b_ in auto_pairs]
         + [(a, b_, "forced") for a, b_ in FORCED_IMPERFECT_PAIRS]
         + [(a, b_, "heldout") for a, b_ in HELDOUT_PAIRS])

streams = []
for a, b_, pair_type in pairs:
    mask_a, mask_b = subj_all == a, subj_all == b_
    streams.append({
        "a": a, "b": b_, "pair_type": pair_type, "pair_id": f"{a}-{b_}",
        "emb": np.concatenate([all_embeddings[mask_a], all_embeddings[mask_b]]),
        "y": np.concatenate([y_all[mask_a], y_all[mask_b]]),
        "drift_point": int(mask_a.sum()),
    })

print(f"\nRunning {len(POLICY_SPECS)} policies x {len(pairs)} pairs x {len(SEEDS)} seeds "
      f"= {len(POLICY_SPECS) * len(pairs) * len(SEEDS)} runs. This takes a while "
      f"(roughly {len(SEEDS)}x a week6 run plus the 6 new pairs) - one line per pair and seed.\n")

results = []
for seed in SEEDS:
    for s in streams:
        line = []
        for policy, tau in POLICY_SPECS:
            row = run_policy(s["emb"], s["y"], s["drift_point"], policy, tau, seed)
            row.update({"seed": seed, "subject_A": s["a"], "subject_B": s["b"],
                        "pair_type": s["pair_type"], "pair_id": s["pair_id"]})
            results.append(row)
            line.append(f"{short_label(policy, tau)}={row['retrain_events']}/{row['post_drift_accuracy']:.3f}")
        print(f"  seed={seed:<5} A={s['a']:2d} B={s['b']:2d} [{s['pair_type']:7s}] "
              f"(retrains/acc)  " + "  ".join(line))

df = pd.DataFrame(results)
per_run_path = BASE_DIR / "week7_multiseed_per_run.csv"
df.to_csv(per_run_path, index=False)

GROUPS = {
    "tuned_12": df.pair_type.isin(["auto", "forced"]),
    "heldout_6": df.pair_type == "heldout",
    "all_18": df.pair_type.notna(),
}


def summarize_group(frame):
    rows = {}
    for label in POLICY_ORDER:
        f = frame[frame.policy == label]
        per_seed = f.groupby("seed")[["retrain_events", "wasted_retrains", "post_drift_accuracy"]].mean()
        per_pair = f.groupby("pair_id")["post_drift_accuracy"].mean()
        rows[label] = {
            "avg_retrains": per_seed.retrain_events.mean(),
            "avg_wasted_retrains": per_seed.wasted_retrains.mean(),
            "avg_post_drift_acc": per_seed.post_drift_accuracy.mean(),
            "acc_std_across_seeds": per_seed.post_drift_accuracy.std(ddof=1) if len(per_seed) > 1 else np.nan,
            "acc_std_across_pairs": per_pair.std(ddof=1) if len(per_pair) > 1 else np.nan,
        }
    s = pd.DataFrame(rows).T
    s["acc_vs_never_retrain"] = s.avg_post_drift_acc - s.loc["never_retrain", "avg_post_drift_acc"]
    return s.round(3)


boot_rng = np.random.default_rng(0)


def paired_compare(frame, policy_x, policy_y):
    cols = ["post_drift_accuracy", "retrain_events"]
    px = frame[frame.policy == policy_x].groupby("pair_id")[cols].mean()
    py = frame[frame.policy == policy_y].groupby("pair_id")[cols].mean().reindex(px.index)
    d = (px.post_drift_accuracy - py.post_drift_accuracy).to_numpy()
    boots = boot_rng.choice(d, size=(N_BOOTSTRAP, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    y_retrains = py.retrain_events.mean()
    return {
        "n_pairs": len(d),
        "mean_acc_diff": d.mean(),
        "ci95_low": lo,
        "ci95_high": hi,
        "pairs_better": int((d > TIE_TOL).sum()),
        "pairs_tied": int((np.abs(d) <= TIE_TOL).sum()),
        "pairs_worse": int((d < -TIE_TOL).sum()),
        "retrain_ratio": px.retrain_events.mean() / y_retrains if y_retrains > 0 else np.nan,
    }


pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)

summaries = {name: summarize_group(df[mask]) for name, mask in GROUPS.items()}
summary_path = BASE_DIR / "week7_multiseed_summary.csv"
pd.concat(summaries).to_csv(summary_path)

gate_labels = [l for l in POLICY_ORDER if l.startswith("cost_gate")]
baselines = ["never_retrain", "every_flag", "fixed_schedule"]
paired_rows = []
for gname, mask in GROUPS.items():
    frame = df[mask]
    for x in gate_labels:
        for y in baselines:
            r = paired_compare(frame, x, y)
            r.update({"group": gname, "gate": x, "versus": y})
            paired_rows.append(r)
paired = pd.DataFrame(paired_rows)[["group", "gate", "versus", "n_pairs", "mean_acc_diff", "ci95_low",
                                    "ci95_high", "pairs_better", "pairs_tied", "pairs_worse", "retrain_ratio"]]
paired_path = BASE_DIR / "week7_paired_vs_baselines.csv"
paired.round(4).to_csv(paired_path, index=False)

print(f"\nSaved per-run results: {per_run_path}")
print(f"Saved summaries:       {summary_path}")
print(f"Saved paired results:  {paired_path}")

for gname in GROUPS:
    n_pairs = df[GROUPS[gname]].pair_id.nunique()
    print(f"\n=== {gname}  ({n_pairs} pairs x {len(SEEDS)} seeds) ===")
    print(summaries[gname])

print(f"\n=== PAIRED COMPARISON for the pre-chosen gate ({FINAL_LABEL}) ===")
print("mean_acc_diff = gate minus baseline, averaged over pairs; ci95 = bootstrap over pairs;")
print(f"better/tied/worse use a +/-{TIE_TOL} tie band; retrain_ratio = gate retrains / baseline retrains.\n")
print(paired[paired.gate == FINAL_LABEL].round(3).to_string(index=False))

# retrains-vs-accuracy picture: tuning pairs vs held-out pairs, error bars = std across seeds
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    for ax, gname in zip(axes, ["tuned_12", "heldout_6"]):
        s = summaries[gname]
        for label, row in s.iterrows():
            yerr = 0 if np.isnan(row.acc_std_across_seeds) else row.acc_std_across_seeds
            ax.errorbar(row.avg_retrains, row.avg_post_drift_acc, yerr=yerr, fmt="o", capsize=3, zorder=3)
            ax.annotate(label, (row.avg_retrains, row.avg_post_drift_acc),
                        textcoords="offset points", xytext=(6, 5), fontsize=8)
        gate = s.loc[gate_labels]
        ax.plot(gate.avg_retrains, gate.avg_post_drift_acc, linestyle="--", alpha=0.5, zorder=2)
        ax.set_title(f"{gname}  (error bars = std across {len(SEEDS)} seeds)")
        ax.set_xlabel("Average retrains per stream (lower = cheaper)")
        ax.set_ylabel("Average post-drift accuracy")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = BASE_DIR / "week7_multiseed_tradeoff.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nSaved trade-off plot to: {plot_path}")
except ImportError:
    print("\n(matplotlib not installed - skipped the plot)")

print("\nHow to read this:")
print("- heldout_6 is the fair test: tau was fixed before these pairs were run.")
print("  If the gate keeps most of every_flag's accuracy there with far fewer")
print("  retrains, the week6 result generalizes. If it falls apart, report that.")
print("- acc_std_across_seeds small = stable run to run. acc_std_across_pairs large")
print("  = the result depends a lot on which subjects you pick (expect this).")
print("- In the paired table, a CI vs every_flag that includes 0 means accuracy is")
print("  not distinguishable from every_flag; the benefit is the retrain_ratio.")
print("  Only 6 held-out pairs, so intervals there are wide - say so in the report.")