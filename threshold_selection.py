"""
threshold_selection.py
------------------------
Both severity CSVs already tell us, for every flag, whether it was a real
detection (samples_after_true_drift >= 0) or a false alarm (negative - DDM
fired before subject B even entered the stream). That's exactly the
information needed to pick tau properly instead of guessing a number.

The idea: sweep candidate thresholds, and for each one check how many real
false alarms would get correctly gated OUT (severity below tau -> don't
retrain) versus how many real detections would get correctly gated IN
(severity at or above tau -> retrain). The best tau is the one that kills
the most false alarms while sacrificing the fewest real detections.

Note on scope: this picks tau based on severity alone. The full cost-benefit
equation from the plan (severity x expected accuracy gain, divided by
energy cost) needs one more number we don't have yet - how much accuracy a
retrain actually recovers - which only comes once the replay buffer and
retraining step exist. So this tau is the right next deliverable, not the
final decision rule; the energy-cost term gets layered on top of it once
we've built and measured that.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

paths = [
    BASE_DIR / "week3_severity_values.csv",
    BASE_DIR / "week3_severity_imperfect_baseline.csv",
]
frames = [pd.read_csv(p) for p in paths if p.exists()]
if not frames:
    raise FileNotFoundError(
        "Couldn't find either severity CSV next to this script - run "
        "severity_extraction.py and severity_baseline_check.py first."
    )

data = pd.concat(frames, ignore_index=True)
data["is_false_alarm"] = data["samples_after_true_drift"] < 0
data["is_real_detection"] = ~data["is_false_alarm"]

print(f"Loaded {len(data)} total flags: "
      f"{data.is_real_detection.sum()} real detections, "
      f"{data.is_false_alarm.sum()} false alarms\n")

candidate_taus = np.round(np.arange(0.0, 1.01, 0.05), 2)

rows = []
for tau in candidate_taus:
    gated_in = data["severity"] >= tau  # what the cost-gate would retrain on

    false_alarms_avoided = (data.is_false_alarm & ~gated_in).sum()
    false_alarms_missed = (data.is_false_alarm & gated_in).sum()  # still retrains on noise
    real_detections_kept = (data.is_real_detection & gated_in).sum()
    real_detections_lost = (data.is_real_detection & ~gated_in).sum()  # drift ignored, bad

    rows.append({
        "tau": tau,
        "false_alarms_avoided": false_alarms_avoided,
        "false_alarms_still_retrained_on": false_alarms_missed,
        "real_detections_kept": real_detections_kept,
        "real_detections_ignored": real_detections_lost,
    })

sweep = pd.DataFrame(rows)
sweep.to_csv(BASE_DIR / "week3_tau_sweep.csv", index=False)
print(sweep.to_string(index=False))

# Pick the smallest tau that avoids every false alarm while keeping the most
# real detections - "smallest" because a lower tau reacts to milder drift
# too, which is preferable as long as it isn't paying for noise.
clean = sweep[sweep.false_alarms_still_retrained_on == 0]
if len(clean):
    best = clean.sort_values("real_detections_kept", ascending=False).iloc[0]
    print(f"\nRecommended tau = {best.tau}")
    print(f"  -> avoids all {int(data.is_false_alarm.sum())} false alarms in this dataset")
    print(f"  -> still retrains on {int(best.real_detections_kept)} of "
          f"{int(data.is_real_detection.sum())} real detections")
else:
    print("\nNo tau in the sweep fully avoids every false alarm - the highest")
    print("severity false alarm is stronger than at least one real detection.")
    print("Look at week3_tau_sweep.csv and pick the best trade-off manually.")

print("\nNext: this tau becomes the retrain trigger. Once the replay buffer")
print("and last-layer retraining exist, we measure the real accuracy gain")
print("from retraining and fold that into the full cost-benefit equation -")
print("this tau is the severity half of that equation, done and dusted.")