import os
from process import process_folder_recursively
from PCA_and_LDA import run_PCA_LDA
from train_model import cross_validate_pca_lda


# Konfigurasjon
INPUT_DIR = r"C:\Users\nicos\OneDrive - NTNU\Documents\GitHub\IT3212\data\mood_detection\images_processed"
PROCESSED_DIR = r"C:\Users\nicos\OneDrive - NTNU\Documents\GitHub\IT3212\data\mood_detection\images_processed_foss"
TEST_DIR = r"C:\Users\nicos\OneDrive - NTNU\Documents\GitHub\IT3212\data\mood_detection\test"
TRAIN_DIR = r"C:\Users\nicos\OneDrive - NTNU\Documents\GitHub\IT3212\data\mood_detection\train"
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

    
        cross_validate_pca_lda(PROCESSED_DIR, k=50, u=5, num_folds=5)

        
    except ValueError as e:
        print(f"En feil oppstod: {e}")