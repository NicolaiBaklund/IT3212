"""
Script to regenerate facial_features.csv with Image_Path column.
Run this in the venv_mediapipe virtual environment.
"""

from preprocess4 import preprocess_dataset, DEFAULT_OUTPUT_FILE

if __name__ == "__main__":
    print("=" * 80)
    print("REGENERATING CSV WITH IMAGE PATHS")
    print("=" * 80)
    print(f"This will overwrite: {DEFAULT_OUTPUT_FILE}")
    print("Make sure you're in the venv_mediapipe environment!")
    print("=" * 80)
    
    # Regenerate CSV with Image_Path column
    df = preprocess_dataset()
    
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

