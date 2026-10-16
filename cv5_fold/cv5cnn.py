"""
cv5_cnn.py  -  Stage 1: subject-wise 5-fold cross-validation of the CNN baseline
=================================================================================
Run from the F_project folder (same place as cnn_baseline.py / eda.py):
    python cv5_cnn.py

WHAT THIS DOES
  Pools all 30 UCI HAR subjects (train + test files together = 10,299 windows),
  splits the SUBJECTS (not the windows) into 5 folds of ~6 people, and for each
  fold trains a fresh CNN on the other ~24 people and tests it on the held-out
  ~6. Reports mean +/- std of accuracy and macro-F1 across the 5 folds.

WHY SUBJECT-WISE
  Consecutive windows overlap by 50% (your EDA: 91.7% of neighbouring windows
  share 64 samples), and one person's windows look alike. A random fold would
  put near-copies of the same window in train and test and inflate accuracy.
  Subject-wise folds mean every test person is a stranger to the model - which
  is exactly the "new user" situation your whole project is about.

WHAT IT KEEPS FOR STAGE 2
  For each fold it saves cv_cache/fold{k}.npz: the 64-dim embeddings of ALL
  windows from THAT fold's CNN, the classifier head (W0, b0), which subjects
  were held out, and the quantization max. cv5_drift.py uses these so every
  drift stream is built only from people that fold's CNN never saw.

SAME RECIPE AS cnn_baseline.py: Conv1D(32,k5)-MaxPool-Conv1D(64,k5)-GAP-
Dense(64)-Dense(6), Adam 1e-3, batch 64, <=80 epochs, early stopping on a
SUBJECT-held-out validation slice, per-channel z-score from training data only.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["PYTHONHASHSEED"] = "42"

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
N_FOLDS = 5
NUM_CLASSES = 6
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cv_cache"
CACHE_DIR.mkdir(exist_ok=True)
ACTIVITIES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
SIGNAL_NAMES = ["body_acc_x", "body_acc_y", "body_acc_z",
                "body_gyro_x", "body_gyro_y", "body_gyro_z",
                "total_acc_x", "total_acc_y", "total_acc_z"]


def find_root():
    """Locate the real 'UCI HAR Dataset' folder (skips .venv and __MACOSX)."""
    for d, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [x for x in dirs if x not in (".venv", "__MACOSX", ".git", "__pycache__")]
        if "activity_labels.txt" in files and "train" in dirs and "test" in dirs:
            return Path(d)
    raise FileNotFoundError("Could not find the UCI HAR Dataset folder under " + str(BASE_DIR))


def load_split(root, split):
    d = root / split
    X = np.stack([pd.read_csv(d / "Inertial Signals" / f"{n}_{split}.txt", sep=r"\s+",
                              header=None).to_numpy() for n in SIGNAL_NAMES], axis=-1)
    y = pd.read_csv(d / f"y_{split}.txt", sep=r"\s+", header=None)[0].to_numpy() - 1
    s = pd.read_csv(d / f"subject_{split}.txt", sep=r"\s+", header=None)[0].to_numpy()
    return X, y, s


def build_model():
    inp = keras.Input(shape=(128, 9), name="signal_window")
    x = layers.Conv1D(32, 5, activation="relu", padding="same")(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)
    emb = layers.Dense(64, activation="relu", name="embedding")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="classifier")(emb)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    root = find_root()
    print("Dataset:", root)
    Xa, ya, sa = load_split(root, "train")
    Xb, yb, sb = load_split(root, "test")
    X, y, subj = np.concatenate([Xa, Xb]), np.concatenate([ya, yb]), np.concatenate([sa, sb])
    print(f"Pooled: {X.shape[0]} windows, {len(np.unique(subj))} subjects")

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows, subj_rows = [], []
    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), int)

    for k, (tr_all, te) in enumerate(skf.split(X, y, groups=subj), start=1):
        # leakage guard: no person on both sides
        assert not set(subj[tr_all]) & set(subj[te]), "subject leakage!"
        test_subjects = sorted(np.unique(subj[te]).tolist())

        # validation slice carved out of the TRAINING people, again by subject
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        a, b = next(gss.split(X[tr_all], y[tr_all], groups=subj[tr_all]))
        tr, va = tr_all[a], tr_all[b]

        mean = X[tr].mean(axis=(0, 1), keepdims=True)           # train-only statistics
        std = X[tr].std(axis=(0, 1), keepdims=True) + 1e-8
        norm = lambda A: (A - mean) / std

        keras.utils.set_random_seed(SEED + k)
        model = build_model()
        model.fit(norm(X[tr]), y[tr], validation_data=(norm(X[va]), y[va]),
                  epochs=80, batch_size=64, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                                           restore_best_weights=True),
                             keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                               patience=6)])

        pred = model.predict(norm(X[te]), verbose=0).argmax(1)
        acc = float(np.mean(pred == y[te]))
        f1 = float(f1_score(y[te], pred, average="macro"))
        total_cm += confusion_matrix(y[te], pred, labels=range(NUM_CLASSES))
        rows.append({"fold": k, "n_test_subjects": len(test_subjects), "n_test_windows": len(te),
                     "test_subjects": " ".join(map(str, test_subjects)),
                     "accuracy": acc, "macro_f1": f1})
        for s in test_subjects:
            m = subj[te] == s
            subj_rows.append({"fold": k, "subject": s, "n_windows": int(m.sum()),
                              "accuracy": float(np.mean(pred[m] == y[te][m]))})
        print(f"Fold {k}: test subjects {test_subjects} -> acc {acc:.3f}  macro-F1 {f1:.3f}")

        # ---- save embeddings of ALL windows from this fold's CNN, for Stage 2
        emb_model = keras.Model(model.input, model.get_layer("embedding").output)
        emb = emb_model.predict(norm(X), verbose=0)
        W0, b0 = model.get_layer("classifier").get_weights()
        is_train = np.isin(subj, np.unique(subj[tr_all]))
        np.savez(CACHE_DIR / f"fold{k}.npz", emb=emb, y=y, subj=subj, is_train=is_train,
                 W0=W0, b0=b0, test_subjects=np.array(test_subjects))

    df = pd.DataFrame(rows)
    df.to_csv(BASE_DIR / "cv5_cnn_folds.csv", index=False)
    pd.DataFrame(subj_rows).to_csv(BASE_DIR / "cv5_cnn_per_subject.csv", index=False)
    pd.DataFrame(total_cm, index=ACTIVITIES, columns=ACTIVITIES).to_csv(BASE_DIR / "cv5_confusion_matrix.csv")

    print("\n" + "=" * 62)
    print(f"{N_FOLDS}-fold subject-wise CV  (each test person unseen in training)")
    print(f"  Accuracy : {df.accuracy.mean():.3f} +/- {df.accuracy.std():.3f}"
          f"   (min {df.accuracy.min():.3f}, max {df.accuracy.max():.3f})")
    print(f"  Macro-F1 : {df.macro_f1.mean():.3f} +/- {df.macro_f1.std():.3f}")
    print("  Compare with your single-split result of 0.925 (official 21/9 split).")
    print("=" * 62)
    worst = pd.DataFrame(subj_rows).sort_values("accuracy").head(5)
    print("Hardest 5 subjects:\n", worst.to_string(index=False))
    print("\nSaved: cv5_cnn_folds.csv, cv5_cnn_per_subject.csv, cv5_confusion_matrix.csv, cv_cache/fold*.npz")
    print("Next: python cv5_drift.py")


if __name__ == "__main__":
    main()