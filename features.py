"""Feature extraction for image similarity search.

Combines two classical, complementary feature types into a single vector:

- **Color histogram** (HSV): captures what colors are present, independent
  of shape or position.
- **HOG** (Histogram of Oriented Gradients): captures shape/edge structure,
  independent of color.

Each sub-vector is L2-normalized on its own before being weighted and
concatenated, so neither feature type dominates just because it happens to
have more dimensions. No training or pretrained weights are required —
these are hand-engineered features computed directly from pixels.

"""

import cv2
import numpy as np
from skimage.feature import hog


def _l2_normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def extract_color_histogram(image_bgr, config):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [config["hist_bins_h"]], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [config["hist_bins_s"]], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [config["hist_bins_v"]], [0, 256]).flatten()
    combined = np.concatenate([h_hist, s_hist, v_hist])
    return _l2_normalize(combined)


def extract_hog_features(image_bgr, config):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=config["hog_orientations"],
        pixels_per_cell=tuple(config["hog_pixels_per_cell"]),
        cells_per_block=tuple(config["hog_cells_per_block"]),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return _l2_normalize(features)


def extract_features(image_bgr, config):
    """Extract the full combined feature vector for one image."""
    size = tuple(config["resize_to"])
    resized = cv2.resize(image_bgr, size)

    parts = []
    if config.get("use_color_histogram", True):
        parts.append(config.get("color_weight", 0.5) * extract_color_histogram(resized, config))
    if config.get("use_hog", True):
        parts.append(config.get("hog_weight", 0.5) * extract_hog_features(resized, config))

    if not parts:
        raise ValueError("At least one of use_color_histogram / use_hog must be enabled.")

    combined = np.concatenate(parts)
    return _l2_normalize(combined)
