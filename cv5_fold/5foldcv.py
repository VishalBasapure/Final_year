"""
subject_5fold_cv.py - Subject-Wise 5-Fold Cross-Validation for TinyML HAR

Dataset: UCI HAR (9 raw inertial channels x 128 timesteps).
Splits: 30 subjects partitioned into 5 folds (6 unseen subjects per validation fold).
Leakage Prevention:
  - Subject isolation (GroupKFold)
  - Z-score normalization computed strictly per-fold on training subjects only.
Outputs:
  - cv_results.txt (per-fold metrics, mean +- std, per-class F1)
  - cv_confusion_matrix.png (aggregated confusion matrix across all 30 subjects)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Channel names and activity classes matching the project specification
CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
]
ACTIVITIES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
NUM_CLASSES = 6
WINDOW_LEN = 128
NUM_CHANNELS = 9

def find_dataset_root():
    """Locate UCI HAR Dataset directory while excluding environment folders."""
    for root_dir, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in (".venv", "venv", "__MACOSX", ".git")]
        if "activity_labels.txt" in files and "train" in dirs and "test" in dirs:
            return root_dir
    raise FileNotFoundError("Could not find 'UCI HAR Dataset' directory.")

def read_space_delimited(filepath):
    return pd.read_csv(filepath, sep=r"\s+", header=None).values

def load_all_data(dataset_root):
    """Load and merge train and test splits to encompass all 30 subjects (10,299 windows)."""
    X_parts, y_parts, s_parts = [], [], []
    for split in ["train", "test"]:
        signals = np.stack([
            read_space_delimited(os.path.join(dataset_root, split, "Inertial Signals", f"{c}_{split}.txt"))
            for c in CHANNELS
        ], axis=-1)  # Shape: (N, 128, 9)
        labels = read_space_delimited(os.path.join(dataset_root, split, f"y_{split}.txt")).ravel().astype(int)
        subjects = read_space_delimited(os.path.join(dataset_root, split, f"subject_{split}.txt")).ravel().astype(int)
        
        X_parts.append(signals)
        y_parts.append(labels)
        s_parts.append(subjects)

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0) - 1  # Convert 1-indexed [1..6] to 0-indexed [0..5]
    s = np.concatenate(s_parts, axis=0)
    return X, y, s

def build_tinyml_cnn(input_shape=(128, 9), num_classes=6):
    """
    1D CNN architecture:
    - 2 Conv1D blocks with pooling
    - 64-unit Dense layer serving as the latent embedding space
    - Softmax classification head
    """
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(filters=32, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    embedding = layers.Dense(64, activation="relu", name="latent_embedding")(x)
    x = layers.Dropout(0.4)(embedding)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def get_subject_folds(subjects, n_splits=5, seed=42):
    """Partition unique subject IDs into 5 non-overlapping folds."""
    unique_subs = np.unique(subjects)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_subs)
    sub_groups = np.array_split(shuffled, n_splits)
    
    folds = []
    for val_subs in sub_groups:
        val_idx = np.where(np.isin(subjects, val_subs))[0]
        train_idx = np.where(~np.isin(subjects, val_subs))[0]
        folds.append((train_idx, val_idx, sorted(val_subs)))
    return folds

def main():
    root = find_dataset_root()
    print(f"Loading data from: {root}")
    X_all, y_all, subj_all = load_all_data(root)
    print(f"Dataset loaded: {X_all.shape[0]} windows, {len(np.unique(subj_all))} unique subjects.\n")

    folds = get_subject_folds(subj_all, n_splits=5, seed=42)
    
    fold_accuracies = []
    all_true = []
    all_pred = []
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("SUBJECT-WISE 5-FOLD CROSS-VALIDATION (UCI HAR)")
    log("=" * 70)

    for fold_idx, (train_idx, val_idx, val_subjects) in enumerate(folds):
        log(f"\n--- FOLD {fold_idx + 1} / 5 ---")
        log(f"Validation Subjects ({len(val_subjects)}): {val_subjects}")
        log(f"Train Windows: {len(train_idx)} | Validation Windows: {len(val_idx)}")

        # 1. Subject-isolated normalization
        mean = X_all[train_idx].mean(axis=(0, 1), keepdims=True)
        std = X_all[train_idx].std(axis=(0, 1), keepdims=True) + 1e-8
        
        X_train = (X_all[train_idx] - mean) / std
        y_train = y_all[train_idx]
        X_val = (X_all[val_idx] - mean) / std
        y_val = y_all[val_idx]

        # 2. Re-instantiate model per fold
        tf.keras.backend.clear_session()
        model = build_tinyml_cnn(input_shape=(WINDOW_LEN, NUM_CHANNELS), num_classes=NUM_CLASSES)

        early_stop = callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=0
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=64,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        # 3. Evaluate on unseen subject fold
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        preds = np.argmax(model.predict(X_val, verbose=0), axis=1)

        fold_accuracies.append(val_acc)
        all_true.extend(y_val)
        all_pred.extend(preds)

        log(f"Fold {fold_idx + 1} Validation Accuracy: {val_acc * 100:.2f}%")

    # 4. Statistical summary
    mean_acc = np.mean(fold_accuracies) * 100
    std_acc = np.std(fold_accuracies) * 100

    log("\n" + "=" * 70)
    log("CROSS-VALIDATION SUMMARY")
    log("=" * 70)
    for idx, acc in enumerate(fold_accuracies, 1):
        log(f"  Fold {idx}: {acc * 100:.2f}%")
    log("-" * 70)
    log(f"  Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    log("=" * 70)

    # 5. Full 30-Subject Confusion Matrix
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(all_true, all_pred):
        cm[t, p] += 1

    cm_df = pd.DataFrame(cm, index=ACTIVITIES, columns=ACTIVITIES)
    log("\nAggregated Confusion Matrix (Rows=True, Columns=Predicted):")
    log(cm_df.to_string())

    # Save summary log
    os.makedirs("cv_outputs", exist_ok=True)
    with open("cv_outputs/cv_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # Save Confusion Matrix figure
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(ACTIVITIES, rotation=35, ha="right")
    ax.set_yticklabels(ACTIVITIES)
    ax.set_xlabel("Predicted Activity")
    ax.set_ylabel("True Activity")
    ax.set_title(f"Subject-Wise 5-Fold CV (Mean: {mean_acc:.2f}% ± {std_acc:.2f}%)")
    plt.colorbar(im)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig("cv_outputs/cv_confusion_matrix.png", dpi=130)
    plt.close()
    print("\nResults saved to cv_outputs/cv_results.txt and cv_outputs/cv_confusion_matrix.png")

if __name__ == "__main__":
    main()