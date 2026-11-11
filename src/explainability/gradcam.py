"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) for Retinal Pathology Localization.

Implementation details (Slide 11):
- Targets the final convolutional layer of EfficientNet-B0 ('top_activation' 7x7x1280).
- Calculates gradients of target severity class score w.r.t. convolutional feature maps.
- Pools gradients to generate importance weights alpha_k.
- Computes rectified linear combination: ReLU(sum_k alpha_k * A^k).
- Normalizes and overlays high-contrast heatmap over fundus photo to highlight
  microaneurysms, hemorrhages, soft/hard exudates, and neovascularization.
"""

from typing import Tuple, Dict, Any, Union, Optional
import os
from pathlib import Path
import cv2
import numpy as np

# Ensure workspace .keras is respected
workspace_keras = Path(__file__).resolve().parents[2] / ".keras"
if workspace_keras.exists():
    os.environ.setdefault("KERAS_HOME", str(workspace_keras))

import tensorflow as tf

try:
    from keras import Model
except ImportError:
    from tensorflow.keras import Model  # type: ignore

from src.data.preprocessing import crop_black_borders, apply_clahe, normalize_image, load_image


class GradCAMVisualizer:
    """
    Grad-CAM engine for clinical explainability of deep retinal screening models.
    """

    def __init__(
        self,
        model: Model,
        target_conv_layer: str = "top_activation",
        backbone_name: str = "efficientnetb0"
    ):
        self.model = model
        self.target_conv_layer = target_conv_layer
        self.backbone_name = backbone_name

        # Resolve backbone and conv layer
        try:
            self.base_model = self.model.get_layer(self.backbone_name)
        except ValueError:
            # Fallback if model was constructed flatly
            self.base_model = self.model

        try:
            self.conv_layer = self.base_model.get_layer(self.target_conv_layer)
        except ValueError:
            # Auto-detect last 4D conv layer
            for layer in reversed(self.base_model.layers):
                if len(layer.output.shape) == 4:
                    self.conv_layer = layer
                    self.target_conv_layer = layer.name
                    break

        # Submodel that extracts convolutional feature maps
        self.feature_extractor = Model(
            inputs=self.base_model.input,
            outputs=[self.conv_layer.output, self.base_model.output]
        )

    def compute_heatmap(
        self,
        preprocessed_tensor: np.ndarray,
        target_index: Optional[int] = None
    ) -> np.ndarray:
        """
        Computes 2D spatial heatmap in [0.0, 1.0].

        Args:
            preprocessed_tensor: Shape (224, 224, 3) or (1, 224, 224, 3) float32.
            target_index: Output index to compute gradients for (e.g. highest positive severity threshold).

        Returns:
            Normalized 2D numpy array of shape (7, 7).
        """
        if len(preprocessed_tensor.shape) == 3:
            inp = tf.convert_to_tensor(np.expand_dims(preprocessed_tensor, axis=0), dtype=tf.float32)
        else:
            inp = tf.convert_to_tensor(preprocessed_tensor, dtype=tf.float32)

        # Scale [0, 1] input to [0, 255] for base_model's internal Rescaling(1/255)
        inp_255 = inp * 255.0 if tf.reduce_max(inp) <= 1.5 else inp

        with tf.GradientTape() as tape:
            conv_features, base_pooled = self.feature_extractor(inp_255, training=False)

            # Route through outer classification head
            gap = self.model.get_layer("gap_pooling")(conv_features)
            dropout = self.model.get_layer("head_dropout")(gap)

            # Determine outer head layer name
            if "ordinal_output" in [l.name for l in self.model.layers]:
                preds = self.model.get_layer("ordinal_output")(dropout)
            else:
                preds = self.model.get_layer("disease_softmax")(dropout)

            # Target class score
            if target_index is None:
                # Default to highest activation
                target_score = tf.reduce_max(preds)
            else:
                idx = min(target_index, preds.shape[-1] - 1)
                target_score = preds[:, idx]

        # Gradients w.r.t. conv feature maps
        grads = tape.gradient(target_score, conv_features)
        if grads is None:
            return np.zeros((conv_features.shape[1], conv_features.shape[2]), dtype=np.float32)

        # Global average pooling of gradients -> weights alpha_k
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weighted combination of feature maps
        cam = tf.reduce_sum(conv_features[0] * pooled_grads, axis=-1)

        # Rectify with ReLU to keep only features with positive influence on prediction
        cam = tf.maximum(cam, 0.0)

        # Normalize to [0.0, 1.0]
        max_val = tf.reduce_max(cam)
        min_val = tf.reduce_min(cam)
        denom = max_val - min_val
        if denom > 1e-8:
            cam = (cam - min_val) / denom
        else:
            cam = tf.zeros_like(cam)

        return cam.numpy().astype(np.float32)

    def explain(
        self,
        image_source: Union[str, np.ndarray],
        target_index: Optional[int] = None,
        alpha: float = 0.40,
        colormap: int = cv2.COLORMAP_JET
    ) -> Dict[str, Any]:
        """
        Generates full explainability bundle:
        1. Resizes heatmap to image dimensions
        2. Blends heatmap over fundus photo
        3. Identifies hot spots / lesion attention coordinates

        Returns:
            Dictionary with heatmap_raw, heatmap_resized, overlay_image, and focus_metric.
        """
        raw_rgb = load_image(image_source)
        cropped = crop_black_borders(raw_rgb)
        enhanced = apply_clahe(cropped, channel_mode="lab")

        # Standard 224x224 input
        resized_224 = cv2.resize(enhanced, (224, 224), interpolation=cv2.INTER_AREA)
        norm_224 = (resized_224.astype(np.float32) / 255.0)

        # Compute 7x7 Grad-CAM
        heatmap_7x7 = self.compute_heatmap(norm_224, target_index=target_index)

        # Upsample to 224x224
        heatmap_224 = cv2.resize(heatmap_7x7, (224, 224), interpolation=cv2.INTER_CUBIC)
        heatmap_224 = np.clip(heatmap_224, 0.0, 1.0)

        # Colorize heatmap
        heatmap_uint8 = (heatmap_224 * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Alpha blend over original resized image
        overlay = cv2.addWeighted(resized_224, 1.0 - alpha, heatmap_colored, alpha, 0)

        # Hotspot fraction (fraction of retina receiving significant attention > 0.6)
        hotspot_fraction = float(np.mean(heatmap_224 > 0.60))

        return {
            "heatmap_raw": heatmap_7x7,
            "heatmap_224": heatmap_224,
            "overlay_image": overlay,
            "preprocessed_image": resized_224,
            "hotspot_fraction": round(hotspot_fraction, 4),
            "target_index": target_index
        }

