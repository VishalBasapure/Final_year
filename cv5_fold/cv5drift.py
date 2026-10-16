"""
cv5_drift.py  -  Stage 2: run your drift pipeline INSIDE the 5 CV folds
========================================================================
Run AFTER cv5_cnn.py (needs cv_cache/fold1..5.npz). Needs only numpy, pandas,
river - no TensorFlow - and takes a few minutes.

THE IDEA
  Fold k has its own CNN that never saw fold k's ~6 test people. So every
  "new user" drift stream for fold k is built from those 6 people only
  (A -> B for every ordered pair, ~30 streams per fold, ~150 in total).
  No stream uses a person the CNN trained on, which was NOT true of some of
  your earlier pairs (e.g. A=1 -> B=16: both were training subjects).

HOW tau IS CHOSEN WITHOUT PEEKING
  For held-out fold k, tau is picked using streams from the OTHER four folds
  (different people, different CNNs), then reported on fold k only.
  Rule: the fewest retrains that still capture >= 90% of the accuracy gain
  that "retrain on every flag" achieves over "no adaptation".

POLICIES (the guide's required baselines + your method + ablation)
  no_adapt, fixed_schedule, every_flag, cost_gate (severity >= tau),
  cost_gate with a float32 (non-quantized) buffer.

SEVERITY NO LONGER PEEKS: baseline = the error rate in the REFERENCE_WINDOW
right before the recent window, not "errors before the true drift point"
(a device cannot know where the drift began).

STILL TRUE (state it in your report): the simulation gives the device true
labels for the recent window when a retrain fires. A real watch has no labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from river import drift

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cv_cache"
N_FOLDS = 5
EMBED_DIM, NUM_CLASSES = 64, 6
SEVERITY_WINDOW = 30
REFERENCE_WINDOW = 100
NEW_DATA_WINDOW = 30
BUFFER_CAP = 50
FIXED_INTERVAL = 100
HOLDOUT_FRAC = 0.2
SEEDS = [42, 43, 44]
TAU_GRID = np.round(np.arange(0.0, 0.61, 0.05), 2)
CAPTURE_TARGET = 0.90


# ----------------------------------------------------------------- building blocks
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def retrain_head(W, b, X, y, rng, epochs=15, lr=1e-3, bs=32):
    """Fine-tune only the 64->6 head (Adam, shuffled each epoch with a seeded rng)."""
    W, b = W.copy(), b.copy()
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    Y = np.eye(NUM_CLASSES)[y]
    t, n = 0, len(y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            g = (softmax(X[idx] @ W + b) - Y[idx]) / len(idx)
            gW, gb = X[idx].T @ g, g.sum(0)
            t += 1
            for p, gr, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W, b


class LatentBuffer:
    """Class-balanced reservoir. quantize=True -> uint8 (your design);
    quantize=False -> float32 (ablation: what does 8-bit cost us?)."""

    def __init__(self, cap, qmax, rng, quantize=True):
        self.cap, self.qmax, self.rng, self.quantize = cap, qmax, rng, quantize
        dt = np.uint8 if quantize else np.float32
        self.slots = {c: np.zeros((cap, EMBED_DIM), dt) for c in range(NUM_CLASSES)}
        self.filled = dict.fromkeys(range(NUM_CLASSES), 0)
        self.seen = dict.fromkeys(range(NUM_CLASSES), 0)

    def _enc(self, e):
        return (np.clip(e / self.qmax, 0, 1) * 255).astype(np.uint8) if self.quantize else e.astype(np.float32)

    def _dec(self, q):
        return q.astype(np.float32) / 255.0 * self.qmax if self.quantize else q

    def add(self, emb, labels):
        for e, c in zip(self._enc(emb), labels):
            c = int(c)
            self.seen[c] += 1
            if self.filled[c] < self.cap:
                self.slots[c][self.filled[c]] = e
                self.filled[c] += 1
            else:
                j = self.rng.integers(0, self.seen[c])
                if j < self.cap:
                    self.slots[c][j] = e

    def get_all(self):
        E, L = [], []
        for c in range(NUM_CLASSES):
            n = self.filled[c]
            if n:
                E.append(self._dec(self.slots[c][:n]))
                L.append(np.full(n, c))
        return np.concatenate(E), np.concatenate(L)


def make_ddm():
    return drift.DDM() if hasattr(drift, "DDM") else drift.binary.DDM()


def severity_now(errors, i):
    r0 = max(0, i - SEVERITY_WINDOW + 1)
    ref = errors[max(0, r0 - REFERENCE_WINDOW):r0]
    if len(ref) < 10:
        return 0.0
    recent, base = np.mean(errors[r0:i + 1]), np.mean(ref)
    return float(np.clip((recent - base) / max(1e-6, 1 - base), 0, 1))


def acc(W, b, X, y):
    return float(np.mean((X @ W + b).argmax(1) == y))


def run_stream(data, a, b_subj, policy, tau=0.15, quantize=True, seed=42):
    emb, y, subj, W0, b0, qmax = (data[k] for k in ("emb", "y", "subj", "W0", "b0", "qmax"))
    rng = np.random.default_rng(seed * 1000 + a * 31 + b_subj)
    ia, ib = np.where(subj == a)[0], np.where(subj == b_subj)[0]
    hold = np.concatenate([rng.choice(ia[y[ia] == c], max(1, int(HOLDOUT_FRAC * (y[ia] == c).sum())),
                                      replace=False) for c in range(NUM_CLASSES) if (y[ia] == c).any()])
    ia = np.setdiff1d(ia, hold)                       # never streamed, never buffered
    Es, ys = np.concatenate([emb[ia], emb[ib]]), np.concatenate([y[ia], y[ib]])
    dp = len(ia)

    W, b = W0.copy(), b0.copy()
    buf = LatentBuffer(BUFFER_CAP, qmax, rng, quantize)
    buf.add(Es[:dp], ys[:dp])
    ddm, errors = make_ddm(), []
    retrains = early = 0
    for i in range(len(ys)):
        err = int((Es[i] @ W + b).argmax() != ys[i])
        errors.append(err)
        ddm.update(err)
        go = False
        if policy == "fixed_schedule":
            go = (i + 1) % FIXED_INTERVAL == 0
        elif policy in ("every_flag", "cost_gate") and getattr(ddm, "drift_detected", False):
            go = True if policy == "every_flag" else severity_now(np.array(errors), i) >= tau
        if go:
            s = max(0, i - NEW_DATA_WINDOW + 1)
            buf.add(Es[s:i + 1], ys[s:i + 1])
            W, b = retrain_head(W, b, *buf.get_all(), rng)
            retrains += 1
            early += int(i < dp)                      # retrain fired before the real drift
    e = np.array(errors)
    return {"retrains": retrains, "retrains_before_drift": early,
            "post_drift_acc": 1 - e[dp:].mean(),
            "forgetting": acc(W0, b0, emb[hold], y[hold]) - acc(W, b, emb[hold], y[hold])}


# ----------------------------------------------------------------- experiment
def load_fold(k):
    z = np.load(CACHE_DIR / f"fold{k}.npz")
    d = {n: z[n] for n in z.files}
    d["qmax"] = float(d["emb"][d["is_train"]].max())      # calibrate on training people only
    ts = sorted(int(s) for s in d["test_subjects"])
    d["streams"] = [(a, b) for a in ts for b in ts if a != b]
    return d


def fold_sweep(d):
    """Per-tau averages for one fold (used to tune tau for OTHER folds)."""
    runs = lambda pol, tau=0.15: pd.DataFrame(
        [run_stream(d, a, b, pol, tau, True, s) for a, b in d["streams"] for s in SEEDS]).mean()
    base, ev = runs("no_adapt"), runs("every_flag")
    rows = []
    for tau in TAU_GRID:
        m = runs("cost_gate", tau)
        rows.append({"tau": tau, "retrains": m.retrains, "post_acc": m.post_drift_acc})
    t = pd.DataFrame(rows)
    t["base_acc"], t["every_acc"], t["every_retrains"] = base.post_drift_acc, ev.post_drift_acc, ev.retrains
    return t


def pick_tau(tables):
    """tables: sweep tables of the folds used for TUNING (not the test fold)."""
    t = pd.concat(tables).groupby("tau").mean().reset_index()
    gain = (t.every_acc - t.base_acc).iloc[0]
    t["capture"] = (t.post_acc - t.base_acc) / gain if gain > 0.005 else np.nan
    ok = t[t.capture >= CAPTURE_TARGET]
    if len(ok):
        return float(ok.sort_values(["retrains", "tau"], ascending=[True, False]).iloc[0].tau), True
    return float(t.sort_values("capture", ascending=False).iloc[0].tau), False


def main():
    missing = [k for k in range(1, N_FOLDS + 1) if not (CACHE_DIR / f"fold{k}.npz").exists()]
    if missing:
        raise SystemExit(f"Missing cv_cache/fold{missing}.npz - run cv5_cnn.py first.")
    folds = {k: load_fold(k) for k in range(1, N_FOLDS + 1)}
    print("Fold -> held-out people (all streams for a fold use ONLY these):")
    for k, d in folds.items():
        print(f"  fold {k}: {sorted(int(s) for s in d['test_subjects'])}  ({len(d['streams'])} streams)")

    print("\nStep 1/2: sweeping tau inside each fold (a few minutes)...")
    sweeps = {}
    for k, d in folds.items():
        sweeps[k] = fold_sweep(d)
        print(f"  fold {k} done")
        sweeps[k].assign(fold=k).to_csv(BASE_DIR / f"cv5_tau_sweep_fold{k}.csv", index=False)

    print("\nStep 2/2: final evaluation, tau tuned on the OTHER four folds...")
    recs, taus = [], {}
    for k, d in folds.items():
        tau, reached = pick_tau([sweeps[j] for j in folds if j != k])
        taus[k] = tau
        print(f"  fold {k}: tau = {tau}" + ("" if reached else "  (no tau reached 90% capture; best-capture used)"))
        configs = [("no_adapt", "no_adapt", tau, True), ("fixed_schedule", "fixed_schedule", tau, True),
                   ("every_flag", "every_flag", tau, True), ("cost_gate", "cost_gate", tau, True),
                   ("cost_gate_old_tau0.15", "cost_gate", 0.15, True),
                   ("cost_gate_FLOAT_buffer", "cost_gate", tau, False)]
        for a, b in d["streams"]:
            for s in SEEDS:
                for name, pol, t_, q in configs:
                    recs.append({"fold": k, "A": a, "B": b, "seed": s, "policy": name,
                                 **run_stream(d, a, b, pol, t_, q, s)})
    df = pd.DataFrame(recs)
    df.to_csv(BASE_DIR / "cv5_drift_per_stream.csv", index=False)

    cols = ["retrains", "retrains_before_drift", "post_drift_acc", "forgetting"]
    per_fold = df.groupby(["policy", "fold"])[cols].mean().reset_index()
    order = ["no_adapt", "fixed_schedule", "every_flag", "cost_gate", "cost_gate_old_tau0.15",
             "cost_gate_FLOAT_buffer"]
    out = pd.DataFrame({c: [f"{per_fold[per_fold.policy == p][c].mean():.3f} +/- "
                            f"{per_fold[per_fold.policy == p][c].std():.3f}" for p in order]
                        for c in cols}, index=order)
    out.to_csv(BASE_DIR / "cv5_drift_summary.csv")
    print("\n=== Mean +/- std over the 5 folds (each fold = mean of its streams x seeds) ===")
    print(out.to_string())

    piv = df.pivot_table(index=["fold", "A", "B", "seed"], columns="policy", values="post_drift_acc")
    print(f"\nPaired post-drift accuracy differences over {len(piv)} stream-runs:")
    for x, y_ in [("every_flag", "no_adapt"), ("cost_gate", "no_adapt"), ("cost_gate", "every_flag"),
                  ("cost_gate", "fixed_schedule"), ("cost_gate_FLOAT_buffer", "cost_gate")]:
        d_ = piv[x] - piv[y_]
        fm = d_.groupby(level="fold").mean()
        print(f"  {x:>22s} - {y_:<14s}: {d_.mean():+.3f}  (fold means {fm.min():+.3f} .. {fm.max():+.3f}, "
              f"better in {(d_ > 0).mean():.0%} of runs)")
    print("\nChosen tau per fold:", taus)
    print("Saved: cv5_drift_summary.csv, cv5_drift_per_stream.csv, cv5_tau_sweep_fold*.csv")


if __name__ == "__main__":
    main()