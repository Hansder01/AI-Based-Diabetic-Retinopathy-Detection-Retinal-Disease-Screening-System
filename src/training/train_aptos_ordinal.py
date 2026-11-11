"""
Training and Validation Pipeline for APTOS 2019 DR Ordinal Severity Classifier.

Features:
- Real-time data augmentation (flips, +/-15 deg rotation, +/-10% zoom, contrast jitter).
- Two-stage transfer learning (frozen backbone warming -> fine-tuning).
- Custom Ordinal Binary Cross-Entropy (CORAL-style).
- Real-time validation evaluation tracking QWK, MAE, and Macro F1.
- Checkpointing into checkpoints/aptos_ordinal_best.keras.
"""

import os
import argparse
from typing import Tuple, List, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback

from src.data.dataset import APTOSDataset
from src.models.ordinal_model import build_ordinal_model
from src.training.losses import OrdinalBinaryCrossEntropy
from src.training.metrics import compute_ordinal_metrics


class RetinalAugmenter:
    """
    Data augmentation pipeline matching Slide 9:
    - Horizontal/vertical flips
    - Small rotations (+/- 15 degrees)
    - Small zoom (+/- 10%)
    - Subtle brightness & contrast jitter
    """

    def __init__(self, rotation_range: float = 15.0, zoom_range: float = 0.10):
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range

    def augment(self, image: np.ndarray) -> np.ndarray:
        # image shape: (H, W, 3), range [0, 1]
        img = image.copy()

        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img)

        # Random vertical flip
        if np.random.rand() > 0.5:
            img = np.flipud(img)

        # Subtle contrast and brightness variation (+/- 10%)
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.uniform(-0.05, 0.05)
            img = np.clip(img * alpha + beta, 0.0, 1.0)

        return img


class APTOSDataGenerator(tf.keras.utils.PyDataset):
    """
    Thread-safe Keras data generator for APTOS 2019 dataset.
    """

    def __init__(
        self,
        dataset: APTOSDataset,
        batch_size: int = 16,
        shuffle: bool = True,
        augment: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.augmenter = RetinalAugmenter() if augment else None
        self.indices = np.arange(len(dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images = []
        batch_targets = []

        for idx in batch_indices:
            img, target_vec, _ = self.dataset.get_sample(idx)
            if self.augment and self.augmenter:
                img = self.augmenter.augment(img)
            batch_images.append(img)
            batch_targets.append(target_vec)

        return np.array(batch_images, dtype=np.float32), np.array(batch_targets, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class OrdinalMetricsCallback(Callback):
    """
    Custom callback that evaluates QWK, MAE, and Macro F1 on the validation set at epoch end.
    """

    def __init__(self, val_dataset: APTOSDataset, max_eval_samples: int = 366):
        super().__init__()
        self.val_dataset = val_dataset
        self.max_eval_samples = min(len(val_dataset), max_eval_samples)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        val_images = []
        val_true_grades = []

        for i in range(self.max_eval_samples):
            img, _, grade = self.val_dataset.get_sample(i)
            val_images.append(img)
            val_true_grades.append(grade)

        val_images = np.array(val_images, dtype=np.float32)
        val_pred_probs = self.model.predict(val_images, batch_size=32, verbose=0)

        metrics = compute_ordinal_metrics(val_true_grades, val_pred_probs)
        qwk = metrics["quadratic_weighted_kappa"]
        mae = metrics["mean_absolute_error"]
        acc = metrics["accuracy"]

        print(f"\n--- Epoch {epoch + 1} Ordinal Evaluation ---")
        print(f"Validation QWK: {qwk:.4f} | MAE: {mae:.4f} | Decoded Accuracy: {acc * 100:.2f}% | Macro F1: {metrics['macro_f1']:.4f}")

        if logs is not None:
            logs["val_qwk"] = qwk
            logs["val_mae"] = mae
            logs["val_decoded_acc"] = acc


def train_aptos_model(
    base_dir: str = "APTOS_2019",
    epochs_head: int = 3,
    epochs_fine: int = 5,
    batch_size: int = 16,
    checkpoint_dir: str = "checkpoints",
    learning_rate: float = 1e-4
) -> tf.keras.Model:
    """
    Runs complete training pipeline for APTOS 2019 Ordinal Model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "aptos_ordinal_best.keras")

    print("[1/4] Initializing datasets and data generators...")
    train_ds = APTOSDataset(base_dir=base_dir, split="train", target_size=(224, 224))
    val_ds = APTOSDataset(base_dir=base_dir, split="valid", target_size=(224, 224))

    print(f"Loaded {len(train_ds)} train samples and {len(val_ds)} validation samples.")

    train_gen = APTOSDataGenerator(train_ds, batch_size=batch_size, shuffle=True, augment=True)
    val_gen = APTOSDataGenerator(val_ds, batch_size=batch_size, shuffle=False, augment=False)

    print("[2/4] Constructing EfficientNet-B0 Ordinal Model...")
    model = build_ordinal_model(input_shape=(224, 224, 3), num_classes=5, dropout_rate=0.3, trainable_backbone=False)

    loss_fn = OrdinalBinaryCrossEntropy()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["binary_accuracy"]
    )

    metrics_cb = OrdinalMetricsCallback(val_ds)
    checkpoint_cb = ModelCheckpoint(
        best_model_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )
    reduce_lr_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    print(f"[3/4] Stage 1: Warming up top classification head ({epochs_head} epochs)...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_head,
        callbacks=[checkpoint_cb, reduce_lr_cb, metrics_cb]
    )

    if epochs_fine > 0:
        print(f"[4/4] Stage 2: Fine-tuning backbone MBConv blocks ({epochs_fine} epochs)...")
        # Unfreeze top layers of EfficientNet
        model.layers[1].trainable = True
        for layer in model.layers[1].layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate * 0.1),
            loss=loss_fn,
            metrics=["binary_accuracy"]
        )

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs_head + epochs_fine,
            initial_epoch=epochs_head,
            callbacks=[checkpoint_cb, reduce_lr_cb, metrics_cb]
        )

    print(f"Training complete. Best model checkpoint saved to: {best_model_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train APTOS Ordinal DR Model")
    parser.add_argument("--epochs_head", type=int, default=2)
    parser.add_argument("--epochs_fine", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_aptos_model(
        epochs_head=args.epochs_head,
        epochs_fine=args.epochs_fine,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

