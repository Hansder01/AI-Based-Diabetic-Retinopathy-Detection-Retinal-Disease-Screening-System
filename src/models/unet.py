"""
U-Net Neural Network Architecture for Retinal Vessel Semantic Segmentation.

Architecture (Slide 7):
- Input: 256x256x3 RGB Fundus Image
- Contracting Path (Encoder):
  - Block 1: 32 filters (128x128 after MaxPool)
  - Block 2: 64 filters (64x64 after MaxPool)
  - Block 3: 128 filters (32x32 after MaxPool)
  - Block 4: 256 filters (16x16 after MaxPool)
  - Bottleneck: 512 filters (8x8 after MaxPool)
- Expanding Path (Decoder):
  - Skip connections concatenating corresponding encoder feature maps
  - Up-convolution (Conv2DTranspose) + Double Conv blocks
- Output: 256x256x1 Binary Vasculature Probability Mask (Sigmoid)
"""

from typing import Tuple, Dict, Any, Union, Optional
import os
import numpy as np
import cv2
import tensorflow as tf

try:
    from keras import layers, Model
except ImportError:
    from tensorflow.keras import layers, Model  # type: ignore

from src.data.preprocessing import preprocess_fundus_image, crop_black_borders, apply_clahe


def _conv_block(inputs: tf.Tensor, num_filters: int, name_prefix: str) -> tf.Tensor:
    """Standard double 3x3 Conv + BatchNorm + ReLU block."""
    x = layers.Conv2D(num_filters, (3, 3), padding="same", name=f"{name_prefix}_conv1")(inputs)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)

    x = layers.Conv2D(num_filters, (3, 3), padding="same", name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)
    return x


def build_unet(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    initial_filters: int = 32
) -> Model:
    """
    Constructs the 4-level U-Net architecture with skip connections.

    Args:
        input_shape: (256, 256, 3)
        initial_filters: Base number of filters (doubles at each encoder stage).

    Returns:
        tf.keras.Model mapping (None, 256, 256, 3) -> (None, 256, 256, 1).
    """
    inputs = layers.Input(shape=input_shape, name="fundus_image_input")

    # --- ENCODER (Contracting Path) ---
    c1 = _conv_block(inputs, initial_filters, "enc1")                    # 256x256
    p1 = layers.MaxPooling2D((2, 2), name="enc1_pool")(c1)               # 128x128

    c2 = _conv_block(p1, initial_filters * 2, "enc2")                    # 128x128
    p2 = layers.MaxPooling2D((2, 2), name="enc2_pool")(c2)               # 64x64

    c3 = _conv_block(p2, initial_filters * 4, "enc3")                    # 64x64
    p3 = layers.MaxPooling2D((2, 2), name="enc3_pool")(c3)               # 32x32

    c4 = _conv_block(p3, initial_filters * 8, "enc4")                    # 32x32
    p4 = layers.MaxPooling2D((2, 2), name="enc4_pool")(c4)               # 16x16

    # --- BOTTLENECK ---
    b = _conv_block(p4, initial_filters * 16, "bottleneck")              # 16x16

    # --- DECODER (Expanding Path with Skip Connections) ---
    u4 = layers.Conv2DTranspose(initial_filters * 8, (2, 2), strides=(2, 2), padding="same", name="dec4_up")(b)
    m4 = layers.Concatenate(name="dec4_concat")([u4, c4])
    c5 = _conv_block(m4, initial_filters * 8, "dec4")

    u3 = layers.Conv2DTranspose(initial_filters * 4, (2, 2), strides=(2, 2), padding="same", name="dec3_up")(c5)
    m3 = layers.Concatenate(name="dec3_concat")([u3, c3])
    c6 = _conv_block(m3, initial_filters * 4, "dec3")

    u2 = layers.Conv2DTranspose(initial_filters * 2, (2, 2), strides=(2, 2), padding="same", name="dec2_up")(c6)
    m2 = layers.Concatenate(name="dec2_concat")([u2, c2])
    c7 = _conv_block(m2, initial_filters * 2, "dec2")

    u1 = layers.Conv2DTranspose(initial_filters, (2, 2), strides=(2, 2), padding="same", name="dec1_up")(c7)
    m1 = layers.Concatenate(name="dec1_concat")([u1, c1])
    c8 = _conv_block(m1, initial_filters, "dec1")

    # --- OUTPUT MASK ---
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", name="vessel_mask_output")(c8)

    model = Model(inputs=inputs, outputs=outputs, name="UNet_RetinalVesselSegmentation")
    return model


class UNetVesselSegmentationWrapper:
    """
    Inference and analysis wrapper for U-Net vessel segmentation.
    Computes vascular caliber, vessel density percentage, and visual overlays.
    """

    def __init__(self, model: Model, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, threshold: float = 0.5):
        model = tf.keras.models.load_model(checkpoint_path, compile=False)
        return cls(model, threshold=threshold)

    def segment(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """
        Runs U-Net inference on a single 256x256x3 normalized image.

        Returns:
            Continuous probability mask of shape (256, 256).
        """
        if len(preprocessed_image.shape) == 3:
            inp = np.expand_dims(preprocessed_image, axis=0)
        else:
            inp = preprocessed_image

        prob_mask = self.model.predict(inp, verbose=0)[0, :, :, 0]
        return prob_mask

    def analyze_vasculature(
        self,
        image_source: Union[str, np.ndarray],
        target_size: Tuple[int, int] = (256, 256)
    ) -> Dict[str, Any]:
        """
        End-to-end vascular analysis pipeline:
        1. Preprocesses image
        2. Generates vessel probability mask and binary mask
        3. Computes vessel density percentage
        4. Generates alpha-blended green overlay visualization

        Returns:
            Dictionary with probability mask, binary mask, vessel density, and overlay.
        """
        if isinstance(image_source, str):
            bgr = cv2.imread(image_source)
            if bgr is None:
                raise FileNotFoundError(f"Could not load image: {image_source}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = image_source

        # Preprocess
        cropped = crop_black_borders(rgb)
        enhanced = apply_clahe(cropped, channel_mode="green")
        resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_AREA)
        norm_img = resized.astype(np.float32) / 255.0

        # Predict
        prob_mask = self.segment(norm_img)
        binary_mask = (prob_mask >= self.threshold).astype(np.uint8)

        # Retinal disc non-black area mask
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        retina_mask = (gray > 15).astype(np.uint8)
        retina_area = int(np.sum(retina_mask))

        # Mask vessel predictions so they only exist within the true retinal disc
        masked_binary = binary_mask * retina_mask
        vessel_pixels = int(np.sum(masked_binary))

        # Vessel density = vessel area / total retinal disc area (strictly in [0.0, 100.0])
        vessel_density = (vessel_pixels / retina_area * 100.0) if retina_area > 0 else 0.0
        vessel_density = float(np.clip(vessel_density, 0.0, 100.0))

        # Generate visual overlay (green highlight on original resized image)
        overlay = resized.copy()
        overlay[masked_binary == 1] = [0, 255, 120]  # Vibrant medical cyan-green
        blended = cv2.addWeighted(resized, 0.65, overlay, 0.35, 0)

        return {
            "probability_mask": prob_mask,
            "binary_mask": masked_binary,
            "vessel_density_percent": round(vessel_density, 2),
            "vessel_pixel_count": vessel_pixels,
            "retina_area_pixels": retina_area,
            "overlay_image": blended,
            "preprocessed_image": resized
        }

