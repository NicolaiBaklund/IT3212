"""
Helper utilities for preprocessing steps used in the LBP pipeline.


Mulige forbedringer: 
- Gjenofør flere prosesseringssteg som kan gjøres i samme funksjon.
- Derav crop, CLAHE, contrast_min.
"""

from typing import Optional, Tuple, Union
from typing import List

import sys
from pathlib import Path

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from image_processing.PCA.alignement import align_face_bgr


def process_single_image(
    image: np.ndarray,
    target_size: int = 256,
    normalize_0_1: bool = True,
    min_conf: float = 0.85,
    detector: Optional[MTCNN] = None,
    return_info: bool = False,
) -> Union[np.ndarray, Tuple[Optional[np.ndarray], dict]]:
    """
    Align, grayscale, and normalise a single in-memory face image.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D grayscale or 3D BGR). Must be a NumPy array.
    target_size : int
        Side length for the aligned output (default 256).
    normalize_0_1 : bool
        If True, convert the grayscale output to float32 in [0, 1].
    min_conf : float
        Minimum detector confidence to accept the detected face.
    detector : MTCNN or None
        Optional shared detector instance for efficiency.
    return_info : bool
        When True, return a tuple of (processed_image_or_None, info_dict).

    Returns
    -------
    np.ndarray or (np.ndarray | None, dict)
        The processed 2D image (grayscale), optionally paired with metadata.
        On detection failure, returns None (and info if requested).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Expected `image` to be a NumPy array.")

    if detector is None:
        detector = MTCNN()

    if image.ndim == 2:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        img_bgr = image
    else:
        raise ValueError("Input image must be 2D grayscale or 3-channel BGR.")

    aligned, ok, info = align_face_bgr(
        img_bgr,
        detector=detector,
        target_size=target_size,
        min_conf=min_conf,
    )

    if not ok or aligned is None:
        if return_info:
            return None, info
        return None

    processed = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    if normalize_0_1:
        processed = processed.astype(np.float32) / 255.0

    if return_info:
        return processed, info
    return processed


def lbp_vector(image: np.ndarray, grid_size: int = 8, bins: int = 256) -> np.ndarray:
    """
    Compute Local Binary Pattern features using a spatial grid of histograms.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (2D array).
    grid_size : int, optional
        Number of grid cells along each axis (default 8 => 8x8 grid).
    bins : int, optional
        Number of histogram bins per cell (default 256).

    Returns
    -------
    np.ndarray
        Concatenated histogram vector of shape (grid_size * grid_size * bins,).
    """
    if image.ndim != 2:
        raise ValueError("LBP expects a 2D grayscale image.")

    img = image.astype(np.float32, copy=False)
    h, w = img.shape

    cell_h = h // grid_size
    cell_w = w // grid_size
    if cell_h == 0 or cell_w == 0:
        raise ValueError("Grid size is too fine for the input image dimensions.")
    counts_y = [cell_h] * grid_size
    counts_x = [cell_w] * grid_size
    for i in range(h - cell_h * grid_size):
        counts_y[i] += 1
    for i in range(w - cell_w * grid_size):
        counts_x[i] += 1

    padded = np.pad(img, 1, mode="reflect")
    center = padded[1:-1, 1:-1]

    codes = np.zeros((h, w), dtype=np.uint8)
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ]

    for bit, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        codes |= ((neighbor >= center) << bit).astype(np.uint8)

    histograms: List[np.ndarray] = []
    start_h = 0
    for gy in range(grid_size):
        end_h = start_h + counts_y[gy]
        start_w = 0
        for gx in range(grid_size):
            end_w = start_w + counts_x[gx]
            cell_codes = codes[start_h:end_h, start_w:end_w]
            hist = np.bincount(cell_codes.ravel(), minlength=bins).astype(np.float32)
            hist /= hist.sum() if hist.sum() > 0 else 1.0
            histograms.append(hist)
            start_w = end_w
        start_h = end_h

    return np.concatenate(histograms)

