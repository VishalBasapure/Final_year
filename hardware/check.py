"""Quick quality check for logger.py recordings.

Usage (from the hardware folder):
    python check_session.py data/raw/*.csv
    python check_session.py data/raw/*20261105*.csv --ref data/raw_old/s01_STANDING_waist_20261020_232223.csv

Checks per file: duration, sample rate, gaps, gravity units, and whether the
motion level fits the activity name (static vs moving).
With --ref (a static file from an earlier session) it also prints, for files of
the same activity, how many degrees the mean gravity direction moved, which is a direct measure of how
much the mount shifted between sessions.
"""
import argparse, re, sys
import numpy as np
import pandas as pd

STATIC = {"SITTING", "STANDING", "LAYING"}
NAME = re.compile(r"^s(\d+)_(.+)_([a-z]+)_\d{8}_\d{6}\.csv$")


def angle(a, b):
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def check(path, ref_dir):
    name = path.replace("\\", "/").split("/")[-1]
    m = NAME.match(name)
    act = m.group(2) if m else "?"
    d = pd.read_csv(path)
    t = d["t_ms"].values
    acc = d[["ax_g", "ay_g", "az_g"]].values
    norm = np.linalg.norm(acc, axis=1)
    dur = (t[-1] - t[0]) / 1000
    rate = (len(t) - 1) / dur
    gaps = int((np.diff(t) > 30).sum())
    mean = acc.mean(axis=0)
    msgs = []
    if not 0.9 < norm.mean() < 1.1:
        msgs.append(f"UNITS: mean |acc| {norm.mean():.2f} g (want 0.9-1.1) - do not use")
    if abs(rate - 50) > 1.5:
        msgs.append(f"RATE {rate:.1f} Hz (want 50)")
    if gaps:
        msgs.append(f"{gaps} gaps >30 ms")
    if dur < 100:
        msgs.append(f"short: {dur:.0f} s")
    s = norm.std()
    if act in STATIC and s > 0.05:
        msgs.append(f"moved a lot for a static activity (|acc| std {s:.3f} g)")
    if act not in STATIC and act != "?" and s < 0.1:
        msgs.append(f"barely moved for {act} (|acc| std {s:.3f} g) - did you really do it?")
    line = f"{name}: {dur:.0f}s {rate:.1f}Hz |acc|={norm.mean():.3f} std={s:.3f}"
    print(("PASS  " if not msgs else "WARN  ") + line)
    for x in msgs:
        print("      -", x)
    return act, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", help="static file from an earlier session")
    a = ap.parse_args()
    ref = None
    if a.ref:
        d = pd.read_csv(a.ref)
        ref = d[["ax_g", "ay_g", "az_g"]].values.mean(axis=0)
        rm = NAME.match(a.ref.replace("\\", "/").split("/")[-1])
        ref_act = rm.group(2) if rm else None
    for f in a.files:
        act, mean = check(f, None)
        if ref is not None and act == ref_act:
            print(f"      gravity direction vs ref ({ref_act}): {angle(mean, ref):.1f} deg")


if __name__ == "__main__":
    sys.exit(main())