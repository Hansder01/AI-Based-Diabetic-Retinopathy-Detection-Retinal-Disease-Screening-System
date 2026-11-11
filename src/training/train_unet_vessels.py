"""
Training and Validation Pipeline for U-Net Retinal Vessel Segmentation.

Features:
- Synchronized image and binary mask data augmentation.
- Hybrid BCE + Soft-Dice Loss (Slide 9).
- Evaluation tracking Dice Coefficient and IoU.
- Checkpointing into checkpoints/unet_vessel_best.keras.
"""

import os
import sys
from pathlib import Path

# Ensure workspace is in sys.path
sys.path.insert(0, os.getcwd())

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback

from src.data.dataset import FIVESDataset
from src.models.unet import build_unet
from src.training.losses import BCEDiceLoss


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """Computes differentiable Dice score metric."""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred > 0.5, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """Computes Intersection over Union (Jaccard Index) metric."""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred > 0.5, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


class SynchronizedSegmentationAugmenter:
    """
    Applies identical spatial transformations to both the RGB fundus photo
    and its paired binary vessel ground truth mask.
    """

    def augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img = image.copy()
        msk = mask.copy()

        # Synchronized horizontal flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)

        # Synchronized vertical flip
        if np.random.rand() > 0.5:
            img = np.flipud(img)
            msk = np.flipud(msk)

        # Subtle contrast shift on image only
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.9, 1.1)
            img = np.clip(img * factor, 0.0, 1.0)

        return img, msk


class FIVESSegmentationGenerator(tf.keras.utils.PyDataset):
    """
    Keras PyDataset generator for FIVES paired vessel segmentation.
    """

    def __init__(
        self,
        dataset: FIVESDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        augment: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Filter samples to only those with valid ground truth masks
        self.dataset = dataset
        self.indices = [i for i, s in enumerate(dataset.samples) if s["mask_path"] is not None]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.augmenter = SynchronizedSegmentationAugmenter() if augment else None

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        images = []
        masks = []

        for idx in batch_idx:
            img, mask = self.dataset.get_segmentation_sample(idx)
            if self.augment and self.augmenter:
                img, mask = self.augmenter.augment(img, mask)
            images.append(img)
            masks.append(mask)

        return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def train_unet_vessels(
    base_dir: str = "FIVES_Data",
    epochs: int = 5,
    batch_size: int = 8,
    checkpoint_dir: str = "checkpoints",
    learning_rate: float = 1e-4
) -> tf.keras.Model:
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "unet_vessel_best.keras")

    print("[1/3] Loading FIVES image and ground truth mask pairs...")
    train_ds = FIVESDataset(base_dir=base_dir, split="train", segmentation_size=(256, 256))
    test_ds = FIVESDataset(base_dir=base_dir, split="test", segmentation_size=(256, 256))

    train_gen = FIVESSegmentationGenerator(train_ds, batch_size=batch_size, shuffle=True, augment=True)
    val_gen = FIVESSegmentationGenerator(test_ds, batch_size=batch_size, shuffle=False, augment=False)
    print(f"Loaded {len(train_gen.indices)} train pairs and {len(val_gen.indices)} validation pairs.")

    print("[2/3] Constructing 4-Level U-Net with Skip Connections...")
    model = build_unet(input_shape=(256, 256, 3), initial_filters=32)

    loss_fn = BCEDiceLoss(bce_weight=0.5, smooth=1.0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[dice_coefficient, iou_score]
    )

    checkpoint_cb = ModelCheckpoint(
        best_model_path,
        monitor="val_dice_coefficient",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    reduce_lr_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    print(f"[3/3] Training U-Net Vessel Segmentation ({epochs} epochs)...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint_cb, reduce_lr_cb]
    )

    print(f"U-Net training complete. Checkpoint saved to: {best_model_path}")
    return model


if __name__ == "__main__":
    train_unet_vessels()

