from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

_DATA_DEFAULT = Path(__file__).with_name("ferdig_lbp_data.npz")

Array = np.ndarray


def load_grouped_splits(
    dataset_path: Path | str = _DATA_DEFAULT,
    test_size: float = 4 / 19,
    val_size: float = 3 / 15,
    random_state: int = 42,
) -> Tuple[Array, Array, Array, Array, Array, Array, np.ndarray]:
    """Load dataset and return grouped train/val/test splits with sizes logged.

    Returns
    -------
    Tuple of:
        X_train: np.ndarray
        y_train: np.ndarray
        X_val: np.ndarray
        y_val: np.ndarray
        X_test: np.ndarray
        y_test: np.ndarray
        class_names: np.ndarray
    """

    dataset_path = Path(dataset_path)
    data = np.load(dataset_path)

    X = data["features"]
    y = data["labels"]
    groups = data["groups"]
    class_names = data["class_names"]

    #X-en du laster inn vil være av typen numpy.ndarray med dimensjoner (antall_bilder, 16384)
    #som inneholder LBP-vektorene for hvert bilde.

    #y_numeric(labels) vil være av typen numpy.ndarray med dimensjoner (antall_bilder,) 
    #som inneholder etikettene for hvert bilde som er numeriske (f.eks. [0, 1, 2, ...])

    #groups-en vil være av typen numpy.ndarray med dimensjoner (antall_bilder,) 
    #som inneholder person-ID-ene for hvert bilde (f.eks. ['person_1', 'person_2', 'person_3', ...])

    #class_names-en vil være av typen numpy.ndarray med dimensjoner (antall_klassener,) 
    #som inneholder klassenavnene for hver etikett.

    
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

    print("Train size:", len(X_train))
    print("Validation size:", len(X_val))
    print("Test size:", len(X_test))

    return X_train, y_train, X_val, y_val, X_test, y_test, class_names

