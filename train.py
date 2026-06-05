"""
CNN Age Detection Model trained on UTKFace dataset.
Predicts age as both regression (exact age) and classification (age group).

Dataset: https://susanqq.github.io/UTKFace/
         Download and place images in: data/UTKFace/

Usage: python train.py
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data/UTKFace"
IMG_SIZE    = (64, 64)
BATCH_SIZE  = 32
EPOCHS      = 20
RANDOM_SEED = 42

AGE_GROUPS = {
    "Child (0-12)":    (0,  12),
    "Teen (13-19)":    (13, 19),
    "Young (20-35)":   (20, 35),
    "Middle (36-55)":  (36, 55),
    "Senior (56+)":    (56, 120),
}

def get_age_group(age: int) -> str:
    for group, (low, high) in AGE_GROUPS.items():
        if low <= age <= high:
            return group
    return "Senior (56+)"

# ── 1. Load UTKFace Dataset ───────────────────────────────────────────────────
def load_dataset(data_dir: str):
    """
    UTKFace filename format: [age]_[gender]_[race]_[date].jpg
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"UTKFace dataset not found at '{data_dir}'.\n"
            "Download from: https://susanqq.github.io/UTKFace/\n"
            "Extract and place images in: data/UTKFace/"
        )

    images, ages, genders, age_groups = [], [], [], []
    files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
    print(f"Found {len(files)} images in {data_dir}")

    for fname in files:
        parts = fname.split("_")
        if len(parts) < 3:
            continue
        try:
            age    = int(parts[0])
            gender = int(parts[1])
        except ValueError:
            continue

        if age < 0 or age > 116:
            continue

        img_path = os.path.join(data_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        ages.append(age)
        genders.append("Male" if gender == 0 else "Female")
        age_groups.append(get_age_group(age))

    print(f"Loaded {len(images)} valid images")
    return np.array(images), np.array(ages), np.array(genders), np.array(age_groups)

print("Loading UTKFace dataset...")
images, ages, genders, age_groups = load_dataset(DATA_DIR)

# Normalize images
images = images.astype("float32") / 255.0

# Encode age groups
le = LabelEncoder()
age_group_encoded = le.fit_transform(age_groups)
num_classes = len(le.classes_)
print(f"Age groups: {list(le.classes_)}")

# ── 2. Split Dataset ──────────────────────────────────────────────────────────
X_train, X_test, y_age_train, y_age_test, y_grp_train, y_grp_test = train_test_split(
    images, ages, age_group_encoded,
    test_size=0.2, random_state=RANDOM_SEED
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── 3. Build CNN Model ────────────────────────────────────────────────────────
def build_cnn(num_classes: int):
    inputs = layers.Input(shape=(*IMG_SIZE, 3))

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Shared dense
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Output 1 — Regression (exact age)
    age_output = layers.Dense(1, name="age_output")(x)

    # Output 2 — Classification (age group)
    group_output = layers.Dense(num_classes, activation="softmax", name="group_output")(x)

    model = models.Model(inputs, [age_output, group_output])
    return model

model = build_cnn(num_classes)
model.compile(
    optimizer="adam",
    loss={"age_output": "mae", "group_output": "sparse_categorical_crossentropy"},
    loss_weights={"age_output": 1.0, "group_output": 0.5},
    metrics={"age_output": "mae", "group_output": "accuracy"},
)
model.summary()

# ── 4. Callbacks ──────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

cbs = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint("outputs/best_model.keras", save_best_only=True, verbose=1),
    callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
]

# ── 5. Train ──────────────────────────────────────────────────────────────────
print("\nTraining CNN...")
history = model.fit(
    X_train,
    {"age_output": y_age_train, "group_output": y_grp_train},
    validation_data=(X_test, {"age_output": y_age_test, "group_output": y_grp_test}),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=cbs,
)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
eval_results = model.evaluate(
    X_test, {"age_output": y_age_test, "group_output": y_grp_test}, verbose=0
)
print(f"Test MAE (Age)          : {eval_results[1]:.2f} years")
print(f"Test Accuracy (Group)   : {eval_results[4]*100:.2f}%")

# ── 7. Plot Training Curves ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["age_output_mae"],     label="Train MAE")
axes[0].plot(history.history["val_age_output_mae"], label="Val MAE")
axes[0].set_title("Age Regression — MAE")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MAE (years)")
axes[0].legend()

axes[1].plot(history.history["group_output_accuracy"],     label="Train Accuracy")
axes[1].plot(history.history["val_group_output_accuracy"], label="Val Accuracy")
axes[1].set_title("Age Group Classification — Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig("plots/training_curves.png", dpi=150)
plt.close()
print("Training curves saved to plots/training_curves.png")

# ── 8. Sample Predictions & Output CSV ───────────────────────────────────────
predictions = model.predict(X_test[:50])
pred_ages   = predictions[0].flatten().astype(int)
pred_groups = le.inverse_transform(np.argmax(predictions[1], axis=1))
true_groups = le.inverse_transform(y_grp_test[:50])

output_df = pd.DataFrame({
    "TrueAge":       y_age_test[:50],
    "PredictedAge":  pred_ages,
    "AgeDiff":       abs(y_age_test[:50] - pred_ages),
    "TrueGroup":     true_groups,
    "PredictedGroup": pred_groups,
    "GroupCorrect":  true_groups == pred_groups,
})
output_df.to_csv("output.csv", index=False)
print("Predictions saved to output.csv")

# ── 9. Save Model Report ──────────────────────────────────────────────────────
with open("model_report.txt", "w") as f:
    f.write("AGE DETECTION CNN MODEL REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset          : UTKFace\n")
    f.write(f"Total Images     : {len(images)}\n")
    f.write(f"Train Size       : {len(X_train)}\n")
    f.write(f"Test Size        : {len(X_test)}\n")
    f.write(f"Image Size       : {IMG_SIZE}\n")
    f.write(f"Epochs Trained   : {len(history.history['age_output_mae'])}\n\n")
    f.write(f"Test MAE (Age)   : {eval_results[1]:.2f} years\n")
    f.write(f"Test Accuracy    : {eval_results[4]*100:.2f}%\n\n")
    f.write("Age Groups:\n")
    for group, (low, high) in AGE_GROUPS.items():
        f.write(f"  {group}: {low}-{high}\n")
print("Model report saved to model_report.txt")
print("\nDone!")
