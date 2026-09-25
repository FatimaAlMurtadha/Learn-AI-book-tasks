"""Notebook cells: train the model on MNIST plus photo-like variations.

Preprocessing alone cannot close the whole gap. MNIST is clean, centred and
written with one kind of stroke; your photos are slightly rotated, slightly
off-centre and written with a different pen. Training on augmented copies of
MNIST teaches the model to ignore exactly those differences, and it is the
single change that helps most after the preprocessing fix.

Paste the cells below into MNIST_mobile_preprocessing.ipynb, after the split
and the /255.0 scaling, replacing the plain `model.fit(X_train, y_train)`.
"""

# --- Cell 1: augmentation helpers -------------------------------------------
import cv2
import numpy as np


def augment_digit(flat_image, rng):
    """Return one photo-like variation of a flat 784 MNIST image (0-1 floats)."""
    image = (flat_image.reshape(28, 28) * 255).astype(np.uint8)

    # Stroke thickness: a different pen, or a different phone resolution.
    choice = rng.integers(0, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    if choice == 1:
        image = cv2.dilate(image, kernel)
    elif choice == 2:
        image = cv2.erode(image, kernel)

    # Rotation and shift: a photo is never perfectly straight or centred.
    angle = rng.uniform(-12, 12)
    scale = rng.uniform(0.9, 1.1)
    matrix = cv2.getRotationMatrix2D((13.5, 13.5), angle, scale)
    matrix[0, 2] += rng.uniform(-1.5, 1.5)
    matrix[1, 2] += rng.uniform(-1.5, 1.5)
    image = cv2.warpAffine(image, matrix, (28, 28), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Soft edges from resampling a photo.
    if rng.random() < 0.5:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    return image.astype(np.float32).reshape(784) / 255.0


def build_augmented_set(X, y, copies=2, seed=42):
    """Original data plus `copies` augmented versions of every image."""
    rng = np.random.default_rng(seed)
    parts_X, parts_y = [X], [y]
    for _ in range(copies):
        parts_X.append(np.stack([augment_digit(row, rng) for row in X]))
        parts_y.append(y)
    return np.concatenate(parts_X), np.concatenate(parts_y)


# --- Cell 2: train on the augmented training set ----------------------------
# Only the training split is augmented. Validation and test stay untouched, so
# the reported accuracy is still comparable to the original model.
X_train_aug, y_train_aug = build_augmented_set(X_train, y_train, copies=2)
print("Augmented training set:", X_train_aug.shape)

extra_trees_clf.fit(X_train_aug, y_train_aug)

val_accuracy = accuracy_score(y_val, extra_trees_clf.predict(X_val))
test_accuracy = accuracy_score(y_test, extra_trees_clf.predict(X_test))
print(f"Validation accuracy: {val_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")


# --- Cell 3: evaluate on your own photos ------------------------------------
# This is the number worth reporting: accuracy on MNIST says little about how
# the pipeline behaves on phone photos.
from pathlib import Path

from preprocessing import predict_digit

own_images = sorted(Path("digits_images").glob("*.jpg"))
correct = 0
for path in own_images:
    true_label = int("".join(c for c in path.stem if c.isdigit())[-1])
    digit, probabilities = predict_digit(extra_trees_clf, path)
    correct += digit == true_label
    print(f"{path.name}: predicted {digit} "
          f"({probabilities[digit] * 100:.1f}%), true {true_label}")

if own_images:
    print(f"\nAccuracy on own photos: {correct}/{len(own_images)}")


# --- Cell 4: save the model and the metadata --------------------------------
import json

import joblib

joblib.dump(extra_trees_clf, "artifacts/final_model.joblib")
Path("artifacts/metadata.json").write_text(json.dumps({
    "model": "ExtraTreesClassifier",
    "n_estimators": 200,
    "random_state": 42,
    "input_shape": [28, 28],
    "features": 784,
    "normalization": "pixel / 255.0",
    "augmentation": "rotation ±12°, scale 0.9-1.1, shift ±1.5px, dilate/erode, blur",
    "test_accuracy": float(test_accuracy),
}, indent=2), encoding="utf-8")