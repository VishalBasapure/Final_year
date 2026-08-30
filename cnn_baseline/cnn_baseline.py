"""
Week 1 - CNN Baseline + Embedding Extractor
============================================
Trains the small Conv1D feature extractor + classifier head on raw
UCI HAR inertial signals, using the OFFICIAL train/test split so the
result is directly comparable to published benchmarks (expect ~90-96%).

This also gives you a reusable embedding extractor - the 64-dim output
right before the final Dense layer. THIS is what gets 8-bit quantized
and stored in the latent replay buffer in later weeks. Everything after
this script (drift detector on CNN predictions, buffer, replay retrain)
builds on top of what's saved here.

Install first (on top of what you already have):
    pip install tensorflow matplotlib

Run this from inside "UCI HAR Dataset/UCI HAR Dataset/" (the real one,
not the __MACOSX copy) - it expects train/ and test/ folders next to it.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")  # no display needed, just saves a PNG
import matplotlib.pyplot as plt

import os
# Has to happen BEFORE tensorflow is imported - oneDNN (the CPU math library
# TF uses under the hood) can add floating-point numbers in a slightly
# different order across runs when using multiple threads, which is enough
# to nudge results even with every seed fixed. This trades a little speed
# for actually reproducible numbers, which matters more here than speed does
# on a 16K-parameter model.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"] = "42"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Fix every source of randomness up front. Without this, weight init and
# batch shuffling differ every run, so "Subject 5's accuracy" isn't a fixed
# number - it's whatever that particular random training happened to land
# on. That made results genuinely incomparable across runs, not just noisy.
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR.parent / "UCI HAR Dataset" / "train"
TEST_DIR = BASE_DIR.parent / "UCI HAR Dataset" / "test"

# Order matters only in that it must be consistent every time you load data -
# this becomes "channel 0..8" for the model.
SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


# ---------------------------------------------------------------------------
# 1. Load raw signals into shape (N, 128, 9)
#    Each of the 9 files has one row per window, 128 numbers per row.
#    Stacking them gives one array where each sample is a (128, 9) block.
# ---------------------------------------------------------------------------
def load_split(split_dir, split_name):
    channels = []
    for name in SIGNAL_NAMES:
        path = split_dir / "Inertial Signals" / f"{name}_{split_name}.txt"
        arr = pd.read_csv(path, sep=r"\s+", header=None).to_numpy()  # (N, 128)
        channels.append(arr)
    X = np.stack(channels, axis=-1)  # (N, 128, 9)

    y = pd.read_csv(split_dir / f"y_{split_name}.txt", sep=r"\s+", header=None)[0].to_numpy()
    subjects = pd.read_csv(split_dir / f"subject_{split_name}.txt", sep=r"\s+", header=None)[0].to_numpy()
    return X, y, subjects


print("Loading raw inertial signals...")
X_train_full, y_train_full, subj_train_full = load_split(TRAIN_DIR, "train")
X_test, y_test, subj_test = load_split(TEST_DIR, "test")

print(f"Train: {X_train_full.shape}, Test: {X_test.shape}")
print(f"Train subjects: {sorted(np.unique(subj_train_full))}")
print(f"Test subjects:  {sorted(np.unique(subj_test))}")

# Labels in the raw files are 1-6, Keras wants 0-5 for sparse_categorical_crossentropy
y_train_full = y_train_full - 1
y_test = y_test - 1
NUM_CLASSES = 6


# ---------------------------------------------------------------------------
# 2. Hold out a validation slice BY SUBJECT, not randomly.
#    Random splitting would let windows from the same subject leak between
#    train and val, which quietly inflates validation accuracy.
# ---------------------------------------------------------------------------
splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(splitter.split(X_train_full, y_train_full, groups=subj_train_full))

X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

print(f"\nTrain: {X_train.shape[0]} windows, Val: {X_val.shape[0]} windows (held out by subject)")


# ---------------------------------------------------------------------------
# 3. Normalize - z-score each channel using TRAIN statistics only.
#    Val/test must never influence these numbers, or you leak information.
# ---------------------------------------------------------------------------
mean = X_train.mean(axis=(0, 1), keepdims=True)  # shape (1, 1, 9)
std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test_norm = (X_test - mean) / std

np.save(BASE_DIR / "norm_mean.npy", mean)
np.save(BASE_DIR / "norm_std.npy", std)


# ---------------------------------------------------------------------------
# 4. Build the model
#    Conv1D(32) -> Conv1D(64) -> GlobalAvgPool -> embedding(64) -> Dense(6)
#    The embedding layer is named "embedding" so we can pull it out below -
#    this is the layer your latent replay buffer will store later.
# ---------------------------------------------------------------------------
def build_model():
    inputs = keras.Input(shape=(128, 9), name="signal_window")

    x = layers.Conv1D(32, kernel_size=5, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)

    embedding = layers.Dense(64, activation="relu", name="embedding")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="classifier")(embedding)

    model = keras.Model(inputs, outputs, name="har_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_model()
model.summary()


# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=64,
    callbacks=callbacks,
    verbose=2,
)


# ---------------------------------------------------------------------------
# 6. Evaluate on the OFFICIAL test set (different subjects than train)
# ---------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test_norm, y_test, verbose=0)
print(f"\n{'=' * 60}")
print(f"TEST ACCURACY (official split, unseen subjects): {test_acc:.3f}")
print(f"{'=' * 60}")

if test_acc >= 0.90:
    print("GO: accuracy is in the expected 90-96% published range. Proceed to Week 2.")
elif test_acc >= 0.85:
    print("BORDERLINE: below target but usable. Try more epochs, or double-check normalization, before moving on.")
else:
    print("NO-GO: too low to trust yet. Check signal loading order and normalization before proceeding.")


# ---------------------------------------------------------------------------
# 7. Sanity-check the embedding space
#    Good embeddings should show visible per-class separation.
#    Silhouette score ranges -1 (bad) to +1 (great clusters); >0.2 is a
#    reasonable sign the embeddings are usable for the replay buffer.
# ---------------------------------------------------------------------------
embedding_extractor = keras.Model(model.input, model.get_layer("embedding").output)
test_embeddings = embedding_extractor.predict(X_test_norm, verbose=0)

sample_n = min(1500, len(test_embeddings))  # silhouette is slow on big N, subsample
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(test_embeddings), sample_n, replace=False)

sil_score = silhouette_score(test_embeddings[sample_idx], y_test[sample_idx])
print(f"\nEmbedding silhouette score (class separability): {sil_score:.3f}")
print("  > 0.2 is a reasonable sign the embeddings are usable for the replay buffer.")

pca = PCA(n_components=2)
emb_2d = pca.fit_transform(test_embeddings[sample_idx])

plt.figure(figsize=(7, 6))
scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_test[sample_idx], cmap="tab10", s=8, alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Activity", loc="best")
plt.title("64-dim embedding space, PCA projected to 2D")
plt.tight_layout()
plt.savefig(BASE_DIR / "embedding_pca_plot.png", dpi=150)
print(f"Saved embedding visualization to: {BASE_DIR / 'embedding_pca_plot.png'}")
print("Open that PNG - you want to see 6 roughly separated color clusters, not one blob.")


# ---------------------------------------------------------------------------
# 8. Save everything for later weeks
# ---------------------------------------------------------------------------
model.save(BASE_DIR / "har_cnn_full.keras")
embedding_extractor.save(BASE_DIR / "har_cnn_embedding_extractor.keras")
print(f"\nSaved full model and embedding extractor to {BASE_DIR}")
print("\nNext step: rerun your drift-detector smoke test, but feed it this model's")
print("predictions instead of the RandomForest's - swap:")
print("    preds = clf.predict(stream_X)")
print("with:")
print("    stream_X_norm = (stream_X - mean) / std")
print("    preds = model.predict(stream_X_norm, verbose=0).argmax(axis=1)")
print("on the strong-drift subject pairs you already found (8->17, 1->25, 5->14...).")
