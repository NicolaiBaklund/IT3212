import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from mtcnn.mtcnn import MTCNN

import importlib.util

# This file now benchmarks several preprocessing variants (including the aligned
# dataset produced elsewhere) using a PCA + RandomForest pipeline.

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure legacy absolute import in PCA module resolves
_imagedata_path = ROOT / "image_processing" / "PCA" / "ImageData.py"
spec = importlib.util.spec_from_file_location("ImageData", _imagedata_path)
if spec and spec.loader:
    _imagedata_module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ImageData", _imagedata_module)
    spec.loader.exec_module(_imagedata_module)  # type: ignore[attr-defined]
else:  # pragma: no cover
    raise ImportError(f"Could not load ImageData module from {_imagedata_path}")

from image_processing.PCA.pca import PCA  # type: ignore
from third.preprocess_helper_func import lbp_vector, process_single_image  # type: ignore
from third.train_test_split import load_grouped_splits  # type: ignore

_BASE_DIR = ROOT
DEFAULT_INPUT_ROOT = _BASE_DIR / "data" / "facial_emotion_recognition"
ALIGNED_IMAGE_ROOT = _BASE_DIR / "data" / "images_aligned"
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
RESIZE_SHAPE = (256, 256)
CROP_BASE = (0.18, 0.10, 0.64, 0.75)  # (x_ratio, y_ratio, width_ratio, height_ratio)
CROP_VARIANTS = [
    ("3: Crop Base", 1.0),
    ("4: Crop +10%", 1.10),
    ("5: Crop -10%", 0.90),
    ("6: Crop -15%", 0.85),
]
ABLATION_OUTPUT_ROOT = _BASE_DIR / "data" / "facial_emotion_ablation"
LBP_GRID_SIZE = 8


def _adjust_crop_params(base: Tuple[float, float, float, float], scale: float) -> Tuple[float, float, float, float]:
    x, y, w, h = base
    cx = x + w / 2.0
    cy = y + h / 2.0

    w_new = min(0.999, w * scale)
    h_new = min(0.999, h * scale)

    x_new = cx - w_new / 2.0
    y_new = cy - h_new / 2.0

    x_new = max(0.0, min(x_new, 1.0 - w_new))
    y_new = max(0.0, min(y_new, 1.0 - h_new))

    return x_new, y_new, w_new, h_new


def _crop_by_ratio(image: np.ndarray, crop: Tuple[float, float, float, float]) -> np.ndarray:
    h, w = image.shape
    x_ratio, y_ratio, width_ratio, height_ratio = crop

    x_start = int(round(x_ratio * w))
    y_start = int(round(y_ratio * h))
    x_end = int(round((x_ratio + width_ratio) * w))
    y_end = int(round((y_ratio + height_ratio) * h))

    x_start = max(0, min(x_start, w - 1))
    y_start = max(0, min(y_start, h - 1))
    x_end = max(x_start + 1, min(x_end, w))
    y_end = max(y_start + 1, min(y_end, h))

    return image[y_start:y_end, x_start:x_end]


def _iter_aligned_entries(root: Path) -> Iterable[Tuple[Path, str, str]]:
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name
        for path in sorted(person_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                label = path.stem
                if label.endswith("_aligned"):
                    label = label[: -len("_aligned")]
                yield path, person_id, label


def collect_aligned_faces(
    input_root: Path = ALIGNED_IMAGE_ROOT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    aligned_images: List[np.ndarray] = []
    labels: List[str] = []
    groups: List[str] = []

    if input_root.exists():
        iterator = _iter_aligned_entries(input_root)
    else:
        iterator = _iter_image_entries(DEFAULT_INPUT_ROOT)

    detector: MTCNN | None = None

    for image_path, person_id, label in iterator:
        if input_root.exists():
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            aligned = image.astype(np.float32) / 255.0
        else:
            if detector is None:
                detector = MTCNN()
            original = cv2.imread(str(image_path))
            if original is None:
                continue
            aligned = process_single_image(
                original,
                target_size=RESIZE_SHAPE[0],
                normalize_0_1=True,
                detector=detector,
            )
            if aligned is None:
                continue

        aligned_images.append(aligned.astype(np.float32))
        labels.append(label)
        groups.append(person_id)

    if not aligned_images:
        raise RuntimeError("No aligned faces were produced for the dataset.")

    aligned_array = np.stack(aligned_images)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    groups_array = np.array(groups)
    class_names = encoder.classes_

    return aligned_array, y, groups_array, class_names


def apply_crop_to_aligned(
    aligned_faces: np.ndarray,
    crop_config: Tuple[float, float, float, float],
    target_size: Tuple[int, int] = RESIZE_SHAPE,
) -> np.ndarray:
    crops: List[np.ndarray] = []
    for face in aligned_faces:
        crop = _crop_by_ratio(face, crop_config)
        resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
        crops.append(resized.astype(np.float32))
    return np.stack(crops)


def make_features_from_crops(
    crops: np.ndarray,
    mode: str = "flatten",
    contrast_min: bool = False,
    clahe: bool = False,
) -> np.ndarray:
    features: List[np.ndarray] = []
    clahe_operator = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None

    for crop in crops:
        img_uint8 = np.clip(crop * 255.0, 0, 255).astype(np.uint8)
        processed = img_uint8

        if contrast_min:
            blurred = cv2.GaussianBlur(processed, (99, 99), 0)
            processed = cv2.divide(processed, blurred, scale=128)

        if clahe_operator is not None:
            processed = clahe_operator.apply(processed)

        if mode == "flatten":
            features.append(processed.astype(np.float32).reshape(-1) / 255.0)
        elif mode == "lbp":
            features.append(lbp_vector(processed))
        else:
            raise ValueError(f"Unsupported feature mode '{mode}'.")

    return np.stack(features)


def _save_variant_example(name: str, image: np.ndarray) -> None:
    ABLATION_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    safe_name = name.lower().replace(" ", "_").replace(":", "_").replace("(", "").replace(")", "").replace("+", "plus")
    path = ABLATION_OUTPUT_ROOT / f"{safe_name}.png"
    image_to_save = np.asarray(image, dtype=np.float32)

    # Handle different input shapes
    if image_to_save.ndim == 1:
        # Try to reshape if it's a flattened image
        total_pixels = image_to_save.size
        # Try to infer square shape
        side = int(np.sqrt(total_pixels))
        if side * side == total_pixels:
            image_to_save = image_to_save.reshape(side, side)
        else:
            raise ValueError(f"Cannot reshape 1D array of size {total_pixels} to 2D image")
    elif image_to_save.ndim == 3:
        if image_to_save.shape[2] == 1:
            image_to_save = image_to_save[:, :, 0]
        elif image_to_save.shape[2] == 3:
            # Convert BGR to grayscale
            image_to_save = cv2.cvtColor(image_to_save.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unsupported 3D image shape: {image_to_save.shape}")
    elif image_to_save.ndim != 2:
        raise ValueError(f"Unsupported image dimensions: {image_to_save.ndim}D, shape: {image_to_save.shape}")

    # Normalize to 0-1 range if needed
    if image_to_save.dtype in (np.float32, np.float64):
        image_min = float(np.min(image_to_save))
        image_max = float(np.max(image_to_save))
        if image_max > image_min:
            if image_max <= 1.0:
                # Already in 0-1 range
                image_to_save = (image_to_save * 255.0).round().astype(np.uint8)
            else:
                # Scale to 0-255
                image_to_save = np.clip((image_to_save - image_min) / (image_max - image_min) * 255.0, 0, 255).round().astype(np.uint8)
        else:
            # All values are the same
            image_to_save = np.full_like(image_to_save, 128, dtype=np.uint8)
    elif image_to_save.dtype not in (np.uint8, np.uint16):
        image_to_save = np.clip(image_to_save, 0, 255).astype(np.uint8)
    else:
        image_to_save = image_to_save.astype(np.uint8)

    # Ensure 2D grayscale image
    if image_to_save.ndim != 2:
        raise ValueError(f"Image must be 2D after processing, got shape: {image_to_save.shape}")

    # Validate image is not empty
    if image_to_save.size == 0:
        raise ValueError(f"Image is empty for variant '{name}'")

    success = cv2.imwrite(str(path), image_to_save)
    if not success:
        raise IOError(f"Failed to save example image to {path}. Image shape: {image_to_save.shape}, dtype: {image_to_save.dtype}")


class _DatasetWrapper:
    """Minimal dataset structure expected by our custom PCA implementation."""

    def __init__(self, data: np.ndarray):
        self.data = data
        self.num_samples, self.num_features = data.shape
        self.mean = np.mean(data, axis=0)
        self.centered_data = data - self.mean


class CustomPCATransformer(BaseEstimator, TransformerMixin):
    """Adaptor so we can place the custom PCA inside a scikit-learn pipeline."""

    def __init__(self, n_components: int | float = 50):
        self.n_components = n_components
        self._pca: PCA | None = None
        self._min: float = 0.0
        self._scale: float = 1.0
        self._components: np.ndarray | None = None
        self._mean_vec: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None):
        X = np.asarray(X, dtype=np.float32)
        self._min = float(np.min(X))
        self._scale = float(np.max(X) - self._min)
        if self._scale > 0:
            X_norm = (X - self._min) / self._scale
        else:
            X_norm = X - self._min
        dataset = _DatasetWrapper(X_norm)

        if isinstance(self.n_components, float):
            max_components = dataset.num_samples
        else:
            max_components = int(self.n_components)
        max_components = max(1, max_components)

        self._pca = PCA(max_components)
        self._pca.fit(dataset)

        total_components = self._pca.components.shape[1]
        explained = getattr(self._pca, "explained_variance_ratio_", None)

        if isinstance(self.n_components, float):
            threshold = min(max(self.n_components, 0.0), 1.0)
            if explained is None:
                keep = total_components
            else:
                cumulative = np.cumsum(explained)
                keep = int(np.searchsorted(cumulative, threshold) + 1)
        else:
            keep = int(self.n_components)

        keep = max(1, min(keep, total_components))

        self._components = self._pca.components[:, :keep]
        self._mean_vec = self._pca.mean
        return self

    def transform(self, X: np.ndarray):
        if self._components is None or self._mean_vec is None:
            raise RuntimeError("CustomPCATransformer must be fitted before calling transform.")
        X = np.asarray(X, dtype=np.float32)
        if self._scale > 0:
            X_norm = (X - self._min) / self._scale
        else:
            X_norm = X - self._min
        return (X_norm - self._mean_vec) @ self._components


def evaluate_random_forest_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int = 50,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Train PCA+RandomForest pipeline and return (accuracy, train_time_seconds)."""

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)

    X_train_combined = np.vstack([X_train, X_val])
    y_train_combined = np.concatenate([y_train, y_val])

    pipeline = Pipeline(
        steps=[
            ("pca", CustomPCATransformer(n_components=0.95)),
            ("model", RandomForestClassifier(n_estimators=200, random_state=random_state)),
        ]
    )

    start = time.perf_counter()
    pipeline.fit(X_train_combined, y_train_combined)
    elapsed = time.perf_counter() - start
    accuracy = float(pipeline.score(X_test, y_test))
    return accuracy, elapsed


def _iter_image_entries(root: Path) -> Iterable[Tuple[Path, str, str]]:
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name
        for path in sorted(person_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                if path.parent == person_dir:
                    label = path.stem
                else:
                    label = path.parent.name
                yield path, person_id, label


def _preprocess_image(image_bgr: np.ndarray, mode: str) -> np.ndarray:
    resized = cv2.resize(image_bgr, RESIZE_SHAPE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    if mode == "grayscale":
        return gray.astype(np.float32)
    if mode == "normalize":
        return gray.astype(np.float32) / 255.0
    if mode == "lbp":
        lbp_vec = lbp_vector(gray)
        return lbp_vec.astype(np.float32)

    raise ValueError(f"Unknown preprocessing mode: {mode}")


def _split_by_groups(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 4 / 19,
    val_size: float = 3 / 15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(X, y, groups))

    X_train_val = X[train_val_idx]
    y_train_val = y[train_val_idx]
    groups_train_val = groups[train_val_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))

    X_train = X_train_val[train_idx]
    y_train = y_train_val[train_idx]
    X_val = X_train_val[val_idx]
    y_val = y_train_val[val_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_flat_variant(mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    labels: List[str] = []
    groups: List[str] = []

    for image_path, person_id, label in _iter_image_entries(DEFAULT_INPUT_ROOT):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        processed = _preprocess_image(image, mode)
        if processed.ndim > 1:
            flattened = processed.flatten().astype(np.float32)
        else:
            flattened = processed.astype(np.float32)
        features.append(flattened)
        labels.append(label)
        groups.append(person_id)

    if not features:
        raise RuntimeError(f"No images processed for mode '{mode}'.")

    X = np.vstack(features)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    groups_array = np.array(groups)

    X_train, y_train, X_val, y_val, X_test, y_test = _split_by_groups(X, y, groups_array)
    class_names = encoder.classes_
    return X_train, y_train, X_val, y_val, X_test, y_test, class_names


def load_aligned_variant() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_grouped_splits()
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    return X_train, y_train, X_val, y_val, X_test, y_test, class_names


@dataclass
class AblationResult:
    step: str
    accuracy: float
    train_time: float


def run_ablation():
    results: List[AblationResult] = []

    # Baseline variants
    baseline_loaders = [
        ("1: Grayscale (baseline)", lambda: load_flat_variant("grayscale")),
        ("2: +Normalize", lambda: load_flat_variant("normalize")),
    ]

    for step_name, loader in baseline_loaders:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = loader()
        # Reshape flattened image back to 2D for visualization
        if X_train[0].ndim == 1:
            example_image = X_train[0].reshape(RESIZE_SHAPE)
        else:
            example_image = X_train[0]
        _save_variant_example(step_name, example_image)
        accuracy, elapsed = evaluate_random_forest_pipeline(X_train, y_train, X_val, y_val, X_test, y_test)
        results.append(AblationResult(step=step_name, accuracy=accuracy, train_time=elapsed))

    # Prepare aligned faces once for downstream cropping
    aligned_faces, aligned_labels, aligned_groups, _ = collect_aligned_faces()

    # Cropping variants on aligned faces
    best_crop_name: str | None = None
    best_crop_accuracy = -1.0
    best_crop_crops: np.ndarray | None = None

    for crop_name, scale in CROP_VARIANTS:
        crop_params = _adjust_crop_params(CROP_BASE, scale)
        crops = apply_crop_to_aligned(aligned_faces, crop_params)
        _save_variant_example(crop_name, crops[0])
        features = make_features_from_crops(crops, mode="flatten")
        X_train, y_train, X_val, y_val, X_test, y_test = _split_by_groups(features, aligned_labels, aligned_groups)
        accuracy, elapsed = evaluate_random_forest_pipeline(X_train, y_train, X_val, y_val, X_test, y_test)

        results.append(AblationResult(step=crop_name, accuracy=accuracy, train_time=elapsed))

        if accuracy > best_crop_accuracy:
            best_crop_accuracy = accuracy
            best_crop_name = crop_name
            best_crop_crops = crops

    if best_crop_crops is None or best_crop_name is None:
        raise RuntimeError("No cropping variant succeeded; cannot proceed with downstream experiments.")

    crops = best_crop_crops
    y = aligned_labels
    groups = aligned_groups

    # Additional processing on best crop
    downstream_variants = [
        (f"{best_crop_name} + LBP", dict(mode="lbp")),
        (
            f"{best_crop_name} + LBP + FlipAug",
            dict(mode="lbp", augment_horizontal_flip=True),
        ),
        (f"{best_crop_name} + ContrastMin", dict(mode="flatten", contrast_min=True)),
        (f"{best_crop_name} + ContrastMin + CLAHE", dict(mode="flatten", contrast_min=True, clahe=True)),
        (f"{best_crop_name} + ContrastMin + CLAHE + LBP", dict(mode="lbp", contrast_min=True, clahe=True)),
    ]

    for variant_name, params in downstream_variants:
        params = params.copy()
        augment_flip = params.pop("augment_horizontal_flip", False)
        features = make_features_from_crops(crops, **params)
        mode = params.get("mode", "flatten")
        contrast_min = params.get("contrast_min", False)
        clahe = params.get("clahe", False)

        y_variant = y
        groups_variant = groups

        if augment_flip:
            flipped_crops = np.flip(crops, axis=2)
            flipped_features = make_features_from_crops(flipped_crops, **params)
            features = np.vstack([features, flipped_features])
            y_variant = np.concatenate([y_variant, y])
            groups_variant = np.concatenate([groups_variant, groups])

        preview = np.clip(crops[0] * 255.0, 0, 255).astype(np.uint8)
        if contrast_min:
            blurred = cv2.GaussianBlur(preview, (99, 99), 0)
            preview = cv2.divide(preview, blurred, scale=128)
        if clahe:
            preview = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(preview)

        flip_preview = None
        if augment_flip:
            flip_preview = np.flip(preview, axis=1)

        if mode == "lbp":
            hist = lbp_vector(preview, grid_size=LBP_GRID_SIZE)
            hist_grid = hist.reshape(LBP_GRID_SIZE, -1)
            hist_vis = cv2.normalize(hist_grid, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if augment_flip and flip_preview is not None:
                flip_hist = lbp_vector(flip_preview, grid_size=LBP_GRID_SIZE)
                flip_hist_grid = flip_hist.reshape(LBP_GRID_SIZE, -1)
                flip_hist_vis = cv2.normalize(flip_hist_grid, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                hist_vis = np.hstack([hist_vis, flip_hist_vis])
            _save_variant_example(variant_name, hist_vis)
        else:
            _save_variant_example(variant_name, preview)

        X_train, y_train, X_val, y_val, X_test, y_test = _split_by_groups(features, y_variant, groups_variant)
        accuracy, elapsed = evaluate_random_forest_pipeline(X_train, y_train, X_val, y_val, X_test, y_test)
        results.append(AblationResult(step=variant_name, accuracy=accuracy, train_time=elapsed))

    print("\nPreprocessing Ablation Study Results")
    print("-" * 70)
    variant_width = 40
    print(f"{'Idx':<4} {'Variant':<{variant_width}} {'Accuracy':>12} {'Train Time (s)':>18}")
    separator_width = 4 + 1 + variant_width + 1 + 12 + 1 + 18
    print("-" * separator_width)
    for idx, result in enumerate(results, start=1):
        print(f"{idx:<4} {result.step:<{variant_width}} {result.accuracy*100:>10.2f}% {result.train_time:>18.2f}")
    print("-" * separator_width)


if __name__ == "__main__":
    run_ablation()

