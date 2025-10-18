import os
from process import process_folder_recursively
from PCA_and_LDA import run_PCA_LDA
from train_model import cross_validate_pca_lda
from  PCA_and_LDA import visualize_fisherfaces
from pathlib import Path

# 1. Definer base-mappen for prosjektet.
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Definer roten til data-mappene (som er data/mood_detection)
# Vi bygger stien fra IT3212 -> data -> mood_detection
DATA_ROOT = BASE_DIR / "data" / "mood_detection"

# 3. Definer de endelige banene ved å bruke DATA_ROOT
INPUT_DIR = DATA_ROOT / "images_processed"
PROCESSED_DIR = DATA_ROOT / "images2-foss"
TEST_DIR = DATA_ROOT / "test"
TRAIN_DIR = DATA_ROOT / "train"


TARGET_SIZE = 256
MIN_CONF = 0.85



if __name__ == "__main__":
    try:
        # Steg 1: Prosesser bilder i mappen rekursivt
        failed_files = process_folder_recursively(
            INPUT_DIR,
            PROCESSED_DIR,
            target_size=TARGET_SIZE,
            min_conf=MIN_CONF,
            do_crop = False,
            do_grayscale= False,
            do_clahe = False,
            do_contrast_min = False,
        )

    
        cross_validate_pca_lda(PROCESSED_DIR, k=70, u=0, num_folds=5)

        
    except ValueError as e:
        print(f"En feil oppstod: {e}")