"""
Phase 1 - ONE DAY SMOKE TEST (not the full rigorous version)
Goal: get a first real f (drift_flag_rate) and w (worth_it_fraction) TODAY.
This is a rough prototype - single run, single subject pair, no multi-seed
averaging. Good enough to know if the idea is even plausible, NOT enough
to cite as final proof. Do the fuller multi-seed version later if this
looks promising.

Install first:
    pip install numpy pandas scikit-learn river
"""

from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from river import drift
except ImportError as exc:  # pragma: no cover - user guidance for setup issues
    raise SystemExit(
        "Missing required packages. Install them with the same Python interpreter "
        "used by your project:\n"
        "python -m pip install numpy pandas scikit-learn river"
    ) from exc

# ---- 1. Load data ----
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "train"

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Training data directory not found at '{DATA_DIR}'. "
        "Run this script from inside the 'UCI HAR Dataset' folder or keep "
        "the dataset folder structure unchanged."
    )

X = pd.read_csv(DATA_DIR / "X_train.txt", sep=r"\s+", header=None)
y = pd.read_csv(DATA_DIR / "y_train.txt", sep=r"\s+", header=None)[0]
subjects = pd.read_csv(DATA_DIR / "subject_train.txt", sep=r"\s+", header=None)[0]

print("Loaded:", X.shape, "samples,", subjects.nunique(), "subjects")

# ---- 2. Pick two subjects: A = "known user", B = "new user" (the drift event) ----
subject_ids = sorted(subjects.unique())
subject_A, subject_B = subject_ids[5], subject_ids[10]  # start simple - first two

mask_A = subjects == subject_A
mask_B = subjects == subject_B

X_A, y_A = X[mask_A].reset_index(drop=True), y[mask_A].reset_index(drop=True)
X_B, y_B = X[mask_B].reset_index(drop=True), y[mask_B].reset_index(drop=True)

print(f"Subject A ({subject_A}): {len(X_A)} samples")
print(f"Subject B ({subject_B}): {len(X_B)} samples")

# ---- 3. Train baseline classifier ONLY on Subject A (this simulates "before drift") ----
X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(
    X_A, y_A, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_A_train, y_A_train)

acc_on_A = clf.score(X_A_test, y_A_test)
acc_on_B = clf.score(X_B, y_B)
print(f"\nAccuracy on Subject A (in-distribution): {acc_on_A:.3f}")
print(f"Accuracy on Subject B (drifted, unseen):  {acc_on_B:.3f}")
print("^ If these two numbers are basically equal, A and B aren't different")
print("  enough to count as 'drift' - pick a different subject pair and rerun.")

# ---- 4. Build the stream: A (pre-drift) then B (post-drift) ----
stream_X = pd.concat([X_A_test, X_B], ignore_index=True)
stream_y = pd.concat([y_A_test, y_B], ignore_index=True)
drift_point = len(X_A_test)  # GROUND TRUTH - this is where real drift happens
print(f"\nGround-truth drift injected at sample index: {drift_point}")

# ---- 5. Run classifier + drift detector sample-by-sample ----
def make_ddm_detector():
    if hasattr(drift, "DDM"):
        return drift.DDM()
    return drift.binary.DDM()


def detector_fired(detector):
    """Support both current and older River drift detector APIs."""
    return bool(getattr(detector, "drift_detected", False))


def run_detector(detector, predictions, truths):
    flags = []
    errors = []

    for i, (pred, true) in enumerate(zip(predictions, truths)):
        is_error = int(pred != true)
        errors.append(is_error)
        detector.update(is_error)
        if detector_fired(detector):
            flags.append(i)

    return flags, errors


def print_detector_report(name, flags, errors, real_drift_point):
    print(f"\n{name}")
    print(f"Total samples in stream: {len(errors)}")
    print(f"Error rate before drift: {np.mean(errors[:real_drift_point]):.3f}")
    print(f"Error rate after drift:  {np.mean(errors[real_drift_point:]):.3f}")
    print(f"Drift flags fired at indices: {flags}")
    print(f"f (drift_flag_rate) = {len(flags) / len(errors):.4f}")

    if flags:
        closest = min(flags, key=lambda idx: abs(idx - real_drift_point))
        print(f"Closest flag to real drift point ({real_drift_point}): "
              f"index {closest}, distance {abs(closest - real_drift_point)} samples")
    else:
        print("NO FLAGS FIRED AT ALL.")


preds = clf.predict(stream_X)

# Sanity check: force obvious label drift in Subject B. If this does not fire,
# the issue is detector wiring/API usage rather than subtle real drift.
rng = np.random.default_rng(42)
shuffled_y_B = y_B.sample(frac=1, random_state=42).reset_index(drop=True)
if shuffled_y_B.equals(y_B.reset_index(drop=True)):
    shuffled_y_B = pd.Series(rng.permutation(y_B.to_numpy()))

sanity_stream_y = pd.concat([y_A_test, shuffled_y_B], ignore_index=True)
sanity_flags, sanity_errors = run_detector(
    drift.PageHinkley(threshold=10, delta=0.001),
    preds,
    sanity_stream_y,
)
print_detector_report(
    "Sanity check: shuffled Subject B labels + sensitive PageHinkley",
    sanity_flags,
    sanity_errors,
    drift_point,
)

ph_flags, ph_errors = run_detector(
    drift.PageHinkley(threshold=10, delta=0.001),
    preds,
    stream_y,
)
print_detector_report(
    "Real A-to-B stream: PageHinkley(threshold=10, delta=0.001)",
    ph_flags,
    ph_errors,
    drift_point,
)

ddm_flags, ddm_errors = run_detector(make_ddm_detector(), preds, stream_y)
print_detector_report(
    "Real A-to-B stream: DDM()",
    ddm_flags,
    ddm_errors,
    drift_point,
)

flags = ph_flags
errors = ph_errors

# ---- 6. Rough severity + worth-it gate ----
WINDOW = 20
worth_it_flags = []
for flag_idx in flags:
    start = max(0, flag_idx - WINDOW)
    recent_error_rate = np.mean(errors[start:flag_idx + 1])
    severity = recent_error_rate  # crude for now - refine later
    is_worth_it = severity > 0.3  # arbitrary threshold - THIS is what you tune later
    worth_it_flags.append(is_worth_it)
    print(f"Flag at {flag_idx}: severity={severity:.2f} -> worth_it={is_worth_it}")

if flags:
    w = sum(worth_it_flags) / len(flags)
    print(f"\nw (worth_it_fraction) = {w:.4f}")
    print("\nPlug these f and w values into the sensitivity table from before")
    print("to see roughly where you land on projected energy savings.")

# ---- 7. Try different subject pairs to see natural drift magnitude ----
print("\nSubject-pair sweep: accuracy drop from A-test to unseen B")
pair_results = []
for candidate_A in subject_ids[:5]:
    candidate_mask_A = subjects == candidate_A
    candidate_X_A = X[candidate_mask_A].reset_index(drop=True)
    candidate_y_A = y[candidate_mask_A].reset_index(drop=True)

    if len(candidate_X_A) < 10:
        continue

    c_X_train, c_X_test, c_y_train, c_y_test = train_test_split(
        candidate_X_A, candidate_y_A, test_size=0.3, random_state=42
    )
    candidate_clf = RandomForestClassifier(n_estimators=50, random_state=42)
    candidate_clf.fit(c_X_train, c_y_train)
    candidate_acc_A = candidate_clf.score(c_X_test, c_y_test)

    for candidate_B in subject_ids:
        if candidate_A == candidate_B:
            continue

        candidate_mask_B = subjects == candidate_B
        candidate_X_B = X[candidate_mask_B].reset_index(drop=True)
        candidate_y_B = y[candidate_mask_B].reset_index(drop=True)
        candidate_acc_B = candidate_clf.score(candidate_X_B, candidate_y_B)
        pair_results.append(
            (candidate_acc_A - candidate_acc_B,
             candidate_A,
             candidate_B,
             candidate_acc_A,
             candidate_acc_B)
        )

for drop, candidate_A, candidate_B, candidate_acc_A, candidate_acc_B in sorted(
    pair_results, reverse=True
)[:10]:
    print(
        f"A={candidate_A:>2} -> B={candidate_B:>2}: "
        f"A_acc={candidate_acc_A:.3f}, B_acc={candidate_acc_B:.3f}, "
        f"drop={drop:.3f}"
    )
