import argparse
import glob
import itertools
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, medfilt, sosfilt, sosfilt_zi

FS = 50
WIN = 128
STEP = 64                 # 50 % overlap, like UCI HAR
TRIM_START_S = 3.0
TRIM_END_S = 1.0
ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING"]
STATIC = ["SITTING", "STANDING", "LAYING"]
FNAME = re.compile(r"s(\d+)_([A-Z_]+?)_([a-z0-9]+)_\d{8}_\d{6}\.csv")


def load_cal(path):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found - using raw accelerometer values.")
        return np.zeros(3), np.ones(3)
    c = json.load(open(path))
    return np.array(c["offset_g"]), np.array(c["gain"])


def load_axes(path):
    if os.path.exists(path):
        return np.array(json.load(open(path))["R"])
    print(f"NOTE: {path} not found - axes left as they are (identity).")
    return np.eye(3)


def lowpass_causal(x, fc):
    sos = butter(3, fc, btype="low", fs=FS, output="sos")
    zi = sosfilt_zi(sos)
    out = np.empty_like(x, dtype=float)
    for c in range(x.shape[1]):
        out[:, c], _ = sosfilt(sos, x[:, c], zi=zi * x[0, c])
    return out


def clean_session(df, offset, gain):
    t = df["t_ms"].to_numpy() / 1000.0
    grid = np.arange(t[0], t[-1], 1.0 / FS)
    acc = np.column_stack([np.interp(grid, t, df[c]) for c in ("ax_g", "ay_g", "az_g")])
    gyr = np.column_stack([np.interp(grid, t, df[c]) for c in ("gx_rads", "gy_rads", "gz_rads")])
    acc = (acc - offset) / gain
    acc = np.column_stack([medfilt(acc[:, i], 3) for i in range(3)])
    gyr = np.column_stack([medfilt(gyr[:, i], 3) for i in range(3)])
    return lowpass_causal(acc, 20.0), lowpass_causal(gyr, 20.0)


def to_channels(acc, gyr, R):
    gravity = lowpass_causal(acc, 0.3)
    body = acc - gravity
    return np.hstack([body @ R.T, gyr @ R.T, acc @ R.T])


def make_windows(sig):
    a = int(TRIM_START_S * FS)
    b = len(sig) - int(TRIM_END_S * FS)
    sig = sig[a:b]
    n = (len(sig) - WIN) // STEP + 1
    if n <= 0:
        return np.empty((0, WIN, 9))
    return np.stack([sig[i * STEP:i * STEP + WIN] for i in range(n)])


def list_sessions(raw_dir, placement=None):
    out = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.csv"))):
        m = FNAME.fullmatch(os.path.basename(path))
        if not m or m.group(2) not in ACTIVITIES:
            print("skipping (name not recognised):", os.path.basename(path))
            continue
        if placement and m.group(3) != placement:
            continue
        out.append((path, int(m.group(1)), m.group(2), m.group(3)))
    return out


def uci_gravity_means(uci_dir):
    label = {"SITTING": 4, "STANDING": 5, "LAYING": 6}
    sums = {k: np.zeros(3) for k in STATIC}
    counts = {k: 0 for k in STATIC}
    for split in ("train", "test"):
        d = Path(uci_dir) / split
        y = pd.read_csv(d / f"y_{split}.txt", sep=r"\s+", header=None)[0].to_numpy()
        tot = np.stack([pd.read_csv(d / "Inertial Signals" / f"total_acc_{ax}_{split}.txt",
                                    sep=r"\s+", header=None).to_numpy() for ax in "xyz"], axis=-1)
        for name, code in label.items():
            sel = tot[y == code]
            sums[name] += sel.sum(axis=(0, 1))
            counts[name] += sel.shape[0] * sel.shape[1]
    return {k: sums[k] / counts[k] for k in STATIC}


def my_gravity_means(raw_dir, offset, gain, placement):
    acc_by = {k: [] for k in STATIC}
    for path, _, activity, _ in list_sessions(raw_dir, placement):
        if activity not in STATIC:
            continue
        acc, _ = clean_session(pd.read_csv(path), offset, gain)
        a = int(TRIM_START_S * FS)
        acc_by[activity].append(acc[a:-int(TRIM_END_S * FS)])
    return {k: np.vstack(v).mean(axis=0) for k, v in acc_by.items() if v}


def angle_deg(u, v):
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def cube_rotations():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[i, p] = s
            if np.linalg.det(M) > 0:
                out.append(M)
    return out


def kabsch(P, Q):
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def cmd_axes(args):
    offset, gain = load_cal(args.cal)
    uci = uci_gravity_means(args.uci)
    mine = my_gravity_means(args.raw, offset, gain, args.placement)
    names = [k for k in STATIC if k in mine]
    if len(names) < 2:
        raise SystemExit("Need waist-worn recordings of at least STANDING and LAYING "
                         "(SITTING too if you can).")
    P = np.array([mine[k] for k in names])
    Q = np.array([uci[k] for k in names])
    print("\nGravity direction (g)      UCI HAR                 your sensor")
    for k in names:
        print(f"  {k:9s}  {np.round(uci[k], 2)}   {np.round(mine[k], 2)}")

    def errors(R):
        return [angle_deg(R @ p, q) for p, q in zip(P, Q)]

    snapped = min(cube_rotations(), key=lambda R: sum(errors(R)))
    free = kabsch(P, Q)
    print("\nBest axis re-labelling (snapped to 90 degree steps):")
    print(snapped.astype(int))
    print("  angle error per activity:", {k: round(float(e), 1) for k, e in zip(names, errors(snapped))}, "deg")
    print("Free rotation (fits tilt too):")
    print("  angle error per activity:", {k: round(float(e), 1) for k, e in zip(names, errors(free))}, "deg")

    R = free if args.free else snapped
    json.dump({"R": R.tolist(), "mode": "free" if args.free else "snapped",
               "errors_deg": {k: float(e) for k, e in zip(names, errors(R))}}, open(args.axes, "w"), indent=2)
    print(f"\nsaved {args.axes} ({'free' if args.free else 'snapped'}).")
    print("Errors under ~15 deg are fine. Much larger means the strap moved between "
          "recordings or the mount is not consistent.")


def cmd_windows(args):
    offset, gain = load_cal(args.cal)
    R = load_axes(args.axes)
    X, y, subj, sess = [], [], [], []
    for path, subject, activity, placement in list_sessions(args.raw, args.placement):
        df = pd.read_csv(path)
        acc, gyr = clean_session(df, offset, gain)
        w = make_windows(to_channels(acc, gyr, R))
        if len(w) == 0:
            print("too short, skipped:", os.path.basename(path))
            continue
        X.append(w)
        y += [ACTIVITIES.index(activity)] * len(w)
        subj += [subject] * len(w)
        sess += [os.path.basename(path)] * len(w)
    if not X:
        raise SystemExit("No usable sessions found.")
    X = np.concatenate(X).astype("float32")
    np.savez_compressed(args.out, X=X, y=np.array(y), subject=np.array(subj), session=np.array(sess))
    print(f"\nsaved {args.out}: {X.shape[0]} windows of shape {X.shape[1:]}")
    table = pd.crosstab(pd.Series(subj, name="subject"),
                        pd.Series([ACTIVITIES[i] for i in y], name="activity"))
    print(table)


def cmd_eval(args):
    from tensorflow import keras
    d = np.load(args.out, allow_pickle=True)
    X, y = d["X"].astype("float32"), d["y"]
    cnn = Path(args.cnn)
    model = keras.models.load_model(cnn / "har_cnn_full.keras", compile=False)
    mean, std = np.load(cnn / "norm_mean.npy"), np.load(cnn / "norm_std.npy")
    pred = model.predict((X - mean) / std, verbose=0).argmax(axis=1)
    print(f"\nOverall accuracy on YOUR sensor data: {(pred == y).mean():.3f}   ({len(y)} windows)")
    names = [ACTIVITIES[i] for i in range(6)]
    cm = pd.crosstab(pd.Series([names[i] for i in y], name="true"),
                     pd.Series([names[i] for i in pred], name="predicted"))
    print(cm)
    for k in sorted(set(y)):
        sel = y == k
        print(f"  {names[k]:20s} {(pred[sel] == k).mean():.3f}  ({sel.sum()} windows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["axes", "windows", "eval"])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--cal", default="accel_cal.json")
    ap.add_argument("--axes", default="axis_map.json")
    ap.add_argument("--out", default="hw_windows.npz")
    ap.add_argument("--placement", default="waist")
    ap.add_argument("--uci", default=str(Path("..") / "UCI HAR Dataset"))
    ap.add_argument("--cnn", default=str(Path("..") / "cnn_baseline"))
    ap.add_argument("--free", action="store_true")
    args = ap.parse_args()
    {"axes": cmd_axes, "windows": cmd_windows, "eval": cmd_eval}[args.command](args)


if __name__ == "__main__":
    main()