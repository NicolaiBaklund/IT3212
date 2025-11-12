"""
Example usage of the preprocessing pipeline with PCA and RandomForest.

This script demonstrates how to:
1. Load preprocessed LBP features with train/val/test splits
2. Apply PCA for dimensionality reduction(0.95 is a good default, this might differ from the original PCA object defult)
3. Train a RandomForest classifier
4. Evaluate the model performance

Pipeline: Load Data → PCA → RandomForest → Evaluate
"""

import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import custom PCA module
import importlib.util

_imagedata_path = ROOT / "image_processing" / "PCA" / "ImageData.py"
spec = importlib.util.spec_from_file_location("ImageData", _imagedata_path)
if spec and spec.loader:
    _imagedata_module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ImageData", _imagedata_module)
    spec.loader.exec_module(_imagedata_module)  # type: ignore[attr-defined]
else:
    raise ImportError(f"Could not load ImageData module from {_imagedata_path}")

from image_processing.PCA.pca import PCA  # type: ignore
from third.train_test_split import load_grouped_splits  # type: ignore


# ============================================================================
# Custom PCA Transformer (from preprocessing_empirical.py)
# ============================================================================

class _DatasetWrapper:
    """Minimal dataset structure expected by our custom PCA implementation."""

    def __init__(self, data: np.ndarray):
        self.data = data
        self.num_samples, self.num_features = data.shape
        self.mean = np.mean(data, axis=0)
        self.centered_data = data - self.mean


class CustomPCATransformer(BaseEstimator, TransformerMixin):
    """
    Adaptor so we can place the custom PCA inside a scikit-learn pipeline.
    
    Parameters:
    -----------
    n_components : int or float
        If int: number of components to keep
        If float (0-1): fraction of variance to retain (e.g., 0.95 = 95% variance)
    """

    def __init__(self, n_components: int | float = 0.95):
        self.n_components = n_components
        self._pca: PCA | None = None
        self._min: float = 0.0
        self._scale: float = 1.0
        self._components: np.ndarray | None = None
        self._mean_vec: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None):
        """Fit the PCA transformer on training data."""
        X = np.asarray(X, dtype=np.float32)
        
        # Normalize to [0, 1] range
        self._min = float(np.min(X))
        self._scale = float(np.max(X) - self._min)
        if self._scale > 0:
            X_norm = (X - self._min) / self._scale
        else:
            X_norm = X - self._min
        
        # Wrap data for custom PCA
        dataset = _DatasetWrapper(X_norm)

        # Determine max components
        if isinstance(self.n_components, float):
            max_components = dataset.num_samples
        else:
            max_components = int(self.n_components)
        max_components = max(1, max_components)

        # Fit PCA
        self._pca = PCA(max_components)
        self._pca.fit(dataset)

        # Determine how many components to keep
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
        """Transform data using fitted PCA."""
        if self._components is None or self._mean_vec is None:
            raise RuntimeError("CustomPCATransformer must be fitted before calling transform.")
        
        X = np.asarray(X, dtype=np.float32)
        
        # Apply same normalization as during fit
        if self._scale > 0:
            X_norm = (X - self._min) / self._scale
        else:
            X_norm = X - self._min
        
        # Project onto principal components
        return (X_norm - self._mean_vec) @ self._components


# ============================================================================
# Main Pipeline Function
# ============================================================================

def train_and_evaluate_pipeline(
    dataset_path: Path | str | None = None,
    n_components: float = 0.95,
    n_estimators: int = 200,
    random_state: int = 42,
) -> Tuple[float, float, Pipeline]:
    """
    Complete pipeline: Load data → PCA → RandomForest → Evaluate.
    
    Parameters:
    -----------
    dataset_path : Path or str, optional
        Path to the preprocessed .npz file. If None, uses default.
    n_components : float, default=0.95
        Fraction of variance to retain in PCA (0-1).
    n_estimators : int, default=200
        Number of trees in RandomForest.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns:
    --------
    test_accuracy : float
        Accuracy on test set
    train_time : float
        Training time in seconds
    pipeline : Pipeline
        Fitted sklearn pipeline
    """
    
    # Step 1: Load preprocessed data with train/val/test splits
    print("=" * 70)
    print("Step 1: Loading preprocessed data...")
    print("=" * 70)
    
    if dataset_path is None:
        dataset_path = Path(__file__).parent / "ferdig_lbp_data.npz"
    
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_grouped_splits(
        dataset_path=dataset_path,
        random_state=random_state
    )
    
    print(f"\nDataset loaded successfully!")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Validation samples: {len(X_val)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Number of classes: {len(class_names)}")
    print(f"  - Class names: {class_names}")
    print(f"  - Feature dimension: {X_train.shape[1]}")
    
    # Step 2: Create pipeline (PCA + RandomForest)
    print("\n" + "=" * 70)
    print("Step 2: Creating pipeline (PCA + RandomForest)...")
    print("=" * 70)
    
    pipeline = Pipeline(
        steps=[
            ("pca", CustomPCATransformer(n_components=n_components)),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,  # Use all CPU cores
            )),
        ]
    )
    
    print(f"  - PCA: Retaining {n_components*100:.1f}% of variance")
    print(f"  - RandomForest: {n_estimators} trees")
    
    # Step 3: Combine train and validation sets for training
    print("\n" + "=" * 70)
    print("Step 3: Training pipeline...")
    print("=" * 70)
    
    X_train_combined = np.vstack([X_train, X_val])
    y_train_combined = np.concatenate([y_train, y_val])
    
    print(f"  - Combined training set: {len(X_train_combined)} samples")
    
    # Train the pipeline
    start_time = time.perf_counter()
    pipeline.fit(X_train_combined, y_train_combined)
    train_time = time.perf_counter() - start_time
    
    print(f"  - Training completed in {train_time:.2f} seconds")
    
    # Get actual number of PCA components used
    pca_transformer = pipeline.named_steps["pca"]
    n_components_used = pca_transformer._components.shape[1]
    print(f"  - PCA reduced features from {X_train.shape[1]} to {n_components_used} components")
    
    # Step 4: Evaluate on test set
    print("\n" + "=" * 70)
    print("Step 4: Evaluating on test set...")
    print("=" * 70)
    
    test_accuracy = pipeline.score(X_test, y_test)
    print(f"  - Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Get predictions for detailed metrics
    y_pred = pipeline.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return test_accuracy, train_time, pipeline


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the complete pipeline.
    
    This demonstrates:
    1. Loading preprocessed LBP features
    2. Applying PCA for dimensionality reduction
    3. Training a RandomForest classifier
    4. Evaluating model performance
    """
    
    print("\n" + "=" * 70)
    print("Facial Emotion Recognition Pipeline")
    print("=" * 70)
    print("\nThis example demonstrates the complete pipeline:")
    print("  1. Load preprocessed LBP features")
    print("  2. Apply PCA (dimensionality reduction)")
    print("  3. Train RandomForest classifier")
    print("  4. Evaluate on test set")
    print()
    
    # Run the pipeline
    accuracy, train_time, pipeline = train_and_evaluate_pipeline(
        dataset_path=None,  # Uses default: third/ferdig_lbp_data.npz
        n_components=0.95,  # Retain 95% of variance
        n_estimators=200,   # Number of trees in RandomForest
        random_state=42,    # For reproducibility
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Pipeline: {pipeline}")
    print("=" * 70)

