# Instructions to Regenerate CSV with Image_Path Column

## Problem
The current `facial_features.csv` doesn't have the `Image_Path` column, which causes the visualization to reconstruct paths. While this works, it's better to have the paths saved directly.

## Solution: Regenerate CSV in venv_mediapipe

### Step 1: Activate the virtual environment
```powershell
.\venv_mediapipe\Scripts\Activate.ps1
```

### Step 2: Run the regeneration script
```powershell
python four/regenerate_csv_with_paths.py
```

Or run directly:
```powershell
python -c "from four.preprocess4 import preprocess_dataset; preprocess_dataset()"
```

### Step 3: Deactivate and return to original environment
```powershell
deactivate
```

## What This Does
- Processes all images in `facial_emotion_recognition_preprocessed2/`
- Extracts geometric features using MediaPipe
- Saves CSV with `Image_Path` column included
- Overwrites the existing `facial_features.csv`

## After Regeneration
The CSV will have the `Image_Path` column, and the visualization will work correctly without needing to reconstruct paths.

