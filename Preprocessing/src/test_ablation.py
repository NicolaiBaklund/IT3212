import sys
from pathlib import Path
import cv2
import numpy as np

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "facial_emotion_recognition"

print(f"ROOT: {ROOT}")
print(f"DATA_ROOT: {DATA_ROOT}")
print(f"DATA_ROOT exists: {DATA_ROOT.exists()}")

if DATA_ROOT.exists():
    person_dirs = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
    print(f"Found {len(person_dirs)} person directories")
    if person_dirs:
        print(f"First person dir: {person_dirs[0]}")
        images = list(person_dirs[0].glob("*.jpg"))
        print(f"Found {len(images)} images in first dir")
        if images:
            img = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
            print(f"Image shape: {img.shape if img is not None else None}")
            print(f"Image dtype: {img.dtype if img is not None else None}")
            print(f"Image min/max: {img.min()}/{img.max() if img is not None else None}")

