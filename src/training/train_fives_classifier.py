"""
Training and Validation Pipeline for FIVES Multi-Disease Pathology Classifier.

Covers:
- AMD, Diabetic Retinopathy, Glaucoma, and Normal Retinal Eyes.
- Real-time data augmentations (flips, +/-15 deg rotation, zoom, contrast jitter).
- EfficientNet-B0 backbone with transfer learning.
- Checkpointing into checkpoints/fives_classifier_best.keras.
"""

import os
import sys
from pathlib import Path

# Ensure workspace is in sys.path
sys.path.insert(0, os.getcwd())

workspace_keras = Path(os.getcwd()) / ".keras"
if workspace_keras.exists():
    os.environ["KERAS_HOME"] = str(workspace_keras)

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from src.data.dataset import FIVESDataset
from src.models.fives_classifier import build_fives_classifier
from src.training.train_aptos_ordinal import RetinalAugmenter


class FIVESClassificationGenerator(tf.keras.utils.PyDataset):
    """
    Keras PyDataset generator for FIVES 4-class disease classification.
    """

    def __init__(
        self,
        dataset: FIVESDataset,
        batch_size: int = 16,
        shuffle: bool = True,
        augment: bool = False,
        num_classes: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.num_classes = num_classes
        self.augmenter = RetinalAugmenter() if augment else None
        self.indices = np.arange(len(dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images = []
        batch_targets = []

        for idx in batch_indices:
            img, class_idx, _ = self.dataset.get_classification_sample(idx)
            if self.augment and self.augmenter:
                img = self.augmenter.augment(img)
            batch_images.append(img)
            # One-hot encode
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[class_idx] = 1.0
            batch_targets.append(one_hot)

        return np.array(batch_images, dtype=np.float32), np.array(batch_targets, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def train_fives_model(
    base_dir: str = "FIVES_Data",
    epochs_head: int = 2,
    epochs_fine: int = 3,
    batch_size: int = 16,
    checkpoint_dir: str = "checkpoints",
    learning_rate: float = 1e-4
) -> tf.keras.Model:
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "fives_classifier_best.keras")

    print("[1/4] Loading FIVES dataset splits...")
    train_ds = FIVESDataset(base_dir=base_dir, split="train", classification_size=(224, 224))
    test_ds = FIVESDataset(base_dir=base_dir, split="test", classification_size=(224, 224))

    print(f"Loaded {len(train_ds)} train samples and {len(test_ds)} test/validation samples.")
    print("Class breakdown (Train):", train_ds.get_class_distribution())

    train_gen = FIVESClassificationGenerator(train_ds, batch_size=batch_size, shuffle=True, augment=True)
    val_gen = FIVESClassificationGenerator(test_ds, batch_size=batch_size, shuffle=False, augment=False)

    print("[2/4] Constructing EfficientNet-B0 4-Class Softmax Model...")
    model = build_fives_classifier(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.3, trainable_backbone=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    checkpoint_cb = ModelCheckpoint(
        best_model_path,
        monitor="val_categorical_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    reduce_lr_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    print(f"[3/4] Stage 1: Warming up top classification head ({epochs_head} epochs)...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_head,
        callbacks=[checkpoint_cb, reduce_lr_cb]
    )

    if epochs_fine > 0:
        print(f"[4/4] Stage 2: Fine-tuning top MBConv blocks ({epochs_fine} epochs)...")
        model.layers[1].trainable = True
        for layer in model.layers[1].layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate * 0.1),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"]
        )

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs_head + epochs_fine,
            initial_epoch=epochs_head,
            callbacks=[checkpoint_cb, reduce_lr_cb]
        )

    print(f"Training complete. Best model checkpoint saved to: {best_model_path}")
    return model


if __name__ == "__main__":
    train_fives_model()

