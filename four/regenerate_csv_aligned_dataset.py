"""
Script to regenerate facial_features.csv with Image_Path column using the aligned dataset.
Run this in the venv_mediapipe virtual environment.

This uses facial_emotion_recognition_aligned/ instead of facial_emotion_recognition_preprocessed2/
"""

from pathlib import Path
from preprocess4 import preprocess_dataset, DEFAULT_OUTPUT_FILE

# Override the dataset folder to use the aligned dataset
_BASE_DIR = Path(__file__).resolve().parent
ALIGNED_DATASET_FOLDER = _BASE_DIR / 'facial_emotion_recognition_aligned'
OUTPUT_FILE = _BASE_DIR / 'facial_features_aligned.csv'

if __name__ == "__main__":
    print("=" * 80)
    print("REGENERATING CSV WITH IMAGE PATHS (ALIGNED DATASET)")
    print("=" * 80)
    print(f"Dataset folder: {ALIGNED_DATASET_FOLDER}")
    print(f"This will create: {OUTPUT_FILE}")
    print("Make sure you're in the venv_mediapipe environment!")
    print("=" * 80)
    
    # Regenerate CSV with Image_Path column using aligned dataset
    df = preprocess_dataset(dataset_folder=ALIGNED_DATASET_FOLDER, output_file=OUTPUT_FILE)
    
    print("\n" + "=" * 80)
    print("CSV REGENERATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    if 'Image_Path' in df.columns:
        print("\n[OK] Image_Path column is now in the CSV!")
        print(f"Sample paths:")
        for i in range(min(5, len(df))):
            print(f"  {df.iloc[i]['Image_Path']}")
    else:
        print("\n[WARNING] Image_Path column not found!")
    print(f"\nCSV saved to: {OUTPUT_FILE}")

