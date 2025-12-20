import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler

# Try to import MediaPipe with helpful error message
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError as e:
    MP_AVAILABLE = False
    print(f"Warning: MediaPipe could not be imported: {e}")
    print("MediaPipe is required for Phase A (preprocessing).")
    print("You may need to resolve protobuf version conflicts between MediaPipe and TensorFlow.")
    print("Try: pip install 'protobuf>=4.25.3,<5.0.0'")

# --- CONFIGURATION ---
_BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_FOLDER = _BASE_DIR / 'facial_emotion_recognition_aligned'
DEFAULT_OUTPUT_FILE = _BASE_DIR / 'facial_features_aligned.csv'

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# MediaPipe Initialization (only if available)
if MP_AVAILABLE:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
else:
    face_mesh = None

def get_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def extract_geometric_features(image_path, label):
    """
    Extract geometric features (FACS proxies) from an image using MediaPipe Face Mesh.
    
    Args:
        image_path: Path to the image file
        label: Emotion label (e.g., 'Anger', 'Happy')
    
    Returns:
        Dictionary of features or None if face not detected
    """
    if not MP_AVAILABLE or face_mesh is None:
        raise RuntimeError("MediaPipe is not available. Cannot extract geometric features.")
    
    image = cv2.imread(str(image_path))
    if image is None: 
        return None
    
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return None  # No face detected

    landmarks = results.multi_face_landmarks[0].landmark

    # --- 1. THE UNIVERSAL RULER (Inter-Ocular Distance) ---
    # Left Pupil: 468, Right Pupil: 473 (Refined landmarks)
    left_pupil = landmarks[468]
    right_pupil = landmarks[473]
    inter_ocular_distance = get_distance(left_pupil, right_pupil)
    
    # Safety check to avoid division by zero
    if inter_ocular_distance == 0: 
        return None
    
    # Helper to get normalized distance (The "Mitigation" Strategy)
    def norm_dist(idx1, idx2):
        dist = get_distance(landmarks[idx1], landmarks[idx2])
        return dist / inter_ocular_distance

    # --- 2. EXTRACT FEATURES (FACS Proxies) ---
    features = {}
    
    # --- MOUTH (Emotions: Happy, Surprise, Sad) ---
    # Mouth Width (Corner to Corner: 61, 291)
    features['Mouth_Width'] = norm_dist(61, 291)
    
    # Mouth Height (Top Lip to Bottom Lip: 13, 14)
    features['Mouth_Height'] = norm_dist(13, 14)
    
    # Mouth Aspect Ratio (Height / Width) - Scale Invariant by definition
    # We add 1e-6 to avoid division by zero
    features['Mouth_Ratio'] = features['Mouth_Height'] / (features['Mouth_Width'] + 1e-6)
    
    # Smile Curvature (Corners vs Lip Center)
    # Check if corners (61, 291) are higher/lower than center (13)
    # Note: In images, Y increases downwards. Lower Y = Higher on face.
    mouth_center_y = landmarks[13].y
    mouth_corners_avg_y = (landmarks[61].y + landmarks[291].y) / 2
    features['Smile_Curve'] = (mouth_corners_avg_y - mouth_center_y) / inter_ocular_distance

    # --- EYES (Emotions: Surprise, Fear, Disgust) ---
    # Left Eye Height (159, 145) / Width (33, 133)
    left_eye_h = get_distance(landmarks[159], landmarks[145])
    left_eye_w = get_distance(landmarks[33], landmarks[133])
    features['Left_Eye_Ratio'] = left_eye_h / (left_eye_w + 1e-6)
    
    # Right Eye Height (386, 374) / Width (362, 263)
    right_eye_h = get_distance(landmarks[386], landmarks[374])
    right_eye_w = get_distance(landmarks[362], landmarks[263])
    features['Right_Eye_Ratio'] = right_eye_h / (right_eye_w + 1e-6)

    # --- BROWS (Emotions: Anger, Sadness) ---
    # Brow to Eye Distance (Left Brow 66 to Eye 159)
    features['Left_Brow_Eye_Dist'] = norm_dist(66, 159)
    
    # Brow to Eye Distance (Right Brow 296 to Eye 386)
    features['Right_Brow_Eye_Dist'] = norm_dist(296, 386)
    
    # Inner Brow Slope (Inner Brow 52 vs Outer Brow 46)
    # Measures the "Sadness" tilt (inverted V)
    features['Left_Brow_Tilt'] = (landmarks[52].y - landmarks[46].y) / inter_ocular_distance
    
    # Brow Squeeze (Distance between brows: 55, 285) - For Anger
    features['Brow_Gap'] = norm_dist(55, 285)

    # --- METADATA ---
    features['Label'] = label  # Keep the label for "Compare and Explain", but drop for training!
    features['Image_Name'] = os.path.basename(image_path)
    
    return features


def _iter_person_dirs(root: Path):
    """Iterate over person directories in the dataset."""
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry


def _iter_images(person_dir: Path):
    """Iterate over image files in a person directory."""
    for path in sorted(person_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def _infer_label(image_path: Path) -> str:
    """Infer emotion label from image filename (e.g., 'Anger.png' -> 'Anger')."""
    return image_path.stem


def preprocess_dataset(
    dataset_folder: Path = DEFAULT_DATASET_FOLDER,
    output_file: Path = DEFAULT_OUTPUT_FILE
) -> pd.DataFrame:
    """
    Phase A: Preprocess images in facial_emotion_recognition_preprocessed2/ folder.
    
    Processes all images using MediaPipe Face Mesh to extract geometric features (FACS proxies).
    The dataset structure should be: person_id/emotion.png (e.g., 0/Anger.png)
    
    Args:
        dataset_folder: Path to the dataset folder containing person subdirectories
        output_file: Path where CSV file will be saved
    
    Returns:
        DataFrame containing all extracted features
    """
    if not dataset_folder.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_folder}")
    
    data = []
    failures = []
    
    print("Starting extraction... this may take a while.")
    
    # Iterate through person directories (0, 1, 2, ..., 18)
    for person_dir in _iter_person_dirs(dataset_folder):
        person_id = person_dir.name
        
        # Iterate through images in person directory
        for image_path in _iter_images(person_dir):
            label = _infer_label(image_path)
            
            try:
                feats = extract_geometric_features(image_path, label)
                if feats:
                    # Add person_id and image path to features
                    feats['Person_ID'] = person_id
                    # Save relative path from dataset folder for easy reconstruction
                    feats['Image_Path'] = str(image_path.relative_to(dataset_folder))
                    data.append(feats)
                else:
                    failures.append((str(image_path), "no_face_detected"))
            except Exception as e:
                failures.append((str(image_path), f"error: {str(e)}"))
                print(f"Error processing {image_path}: {e}")
    
    # Save to CSV
    if not data:
        raise RuntimeError("No features extracted. Check dataset folder and face detection.")
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    print(f"Done! Extracted features for {len(df)} faces. Saved to {output_file}")
    if failures:
        print(f"Failed images: {len(failures)}")
        for path, reason in failures[:10]:
            print(f"  - {reason}: {path}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more.")
    
    return df


def load_and_prepare_data(
    csv_path: Path = DEFAULT_OUTPUT_FILE,
    emotions: List[str] = None,
    num_randoms: int = 0
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Phase B: Load CSV with geometric data, apply StandardScaler, and return selected samples.
    
    This function:
    1. Loads the CSV file with geometric features
    2. Applies StandardScaler to normalize features (crucial: MAR might be 0.5, while 
       Pixel_Distance might be 50 - they need to be scaled to the same range)
    3. Returns all samples from specified emotion groups + num_randoms random samples 
       from other emotions
    
    Args:
        csv_path: Path to the CSV file created by preprocess_dataset()
        emotions: List of emotion labels to include all samples from (e.g., ['Anger', 'Happy'])
                 If None, includes all emotions
        num_randoms: Number of random samples to select from emotions NOT in the emotions list
    
    Returns:
        Tuple of (features_scaled, labels, scaler, selected_indices) where:
        - features_scaled: numpy array of scaled features (n_samples, n_features)
        - labels: numpy array of emotion labels (n_samples,)
        - scaler: Fitted StandardScaler object
        - selected_indices: numpy array of indices in original CSV that were selected
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}. Run preprocess_dataset() first.")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Get feature columns (exclude metadata columns)
    feature_columns = [col for col in df.columns 
                      if col not in ['Label', 'Image_Name', 'Person_ID', 'Image_Path']]
    
    if not feature_columns:
        raise ValueError("No feature columns found in CSV. Expected columns like 'Mouth_Width', 'Mouth_Height', etc.")
    
    # Extract features and labels
    X = df[feature_columns].values
    y = df['Label'].values
    
    # Apply StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Select samples based on emotions and num_randoms
    if emotions is None:
        # Include all samples
        selected_indices = np.arange(len(df))
        emotion_indices = selected_indices
    else:
        # Normalize emotion names (handle case differences)
        emotions_normalized = [e.lower() for e in emotions]
        y_normalized = [label.lower() for label in y]
        
        # Get indices of samples from specified emotions
        emotion_indices = []
        for i, label in enumerate(y_normalized):
            if label in emotions_normalized:
                emotion_indices.append(i)
        
        emotion_indices = np.array(emotion_indices)
        
        # Get indices of samples NOT in specified emotions
        other_indices = [i for i in range(len(df)) if i not in emotion_indices]
        
        # Select random samples from other emotions
        if num_randoms > 0 and len(other_indices) > 0:
            num_randoms = min(num_randoms, len(other_indices))
            np.random.seed(42)  # For reproducibility
            random_indices = np.random.choice(other_indices, size=num_randoms, replace=False)
            selected_indices = np.concatenate([emotion_indices, random_indices])
        else:
            selected_indices = emotion_indices
    
    # Return selected samples
    X_selected = X_scaled[selected_indices]
    y_selected = y[selected_indices]
    
    print(f"Selected {len(X_selected)} samples:")
    if emotions:
        emotion_counts = pd.Series(y_selected).value_counts()
        print(f"  From specified emotions {emotions}: {len(emotion_indices)} samples")
        if num_randoms > 0:
            print(f"  Random samples from other emotions: {num_randoms} samples")
        print(f"  Emotion distribution:")
        for emotion, count in emotion_counts.items():
            print(f"    {emotion}: {count}")
    else:
        emotion_counts = pd.Series(y_selected).value_counts()
        print(f"  All samples included. Emotion distribution:")
        for emotion, count in emotion_counts.items():
            print(f"    {emotion}: {count}")
    
    return X_selected, y_selected, scaler, selected_indices


if __name__ == "__main__":
    # Phase A: Preprocess dataset
    if MP_AVAILABLE:
        print("=" * 60)
        print("PHASE A: Preprocessing dataset")
        print("=" * 60)
        try:
            df = preprocess_dataset()
        except Exception as e:
            print(f"Error in Phase A: {e}")
            print("Skipping Phase A. If CSV already exists, Phase B can still run.")
            df = None
    else:
        print("=" * 60)
        print("PHASE A: Skipped (MediaPipe not available)")
        print("=" * 60)
        df = None
    
    # Phase B: Load and prepare data for clustering
    print("\n" + "=" * 60)
    print("PHASE B: Loading and preparing data")
    print("=" * 60)
    
    # Check if CSV exists
    if DEFAULT_OUTPUT_FILE.exists():
        # Example: Get all Anger and Happy samples + 50 random samples from others
        try:
            X_scaled, y_labels, scaler, selected_indices = load_and_prepare_data(
                emotions=['Anger', 'Happy'],
                num_randoms=50
            )
            
            print(f"\nFinal dataset shape: {X_scaled.shape}")
            print(f"Number of features: {X_scaled.shape[1]}")
            if DEFAULT_OUTPUT_FILE.exists():
                feature_cols = [col for col in pd.read_csv(DEFAULT_OUTPUT_FILE).columns 
                              if col not in ['Label', 'Image_Name', 'Person_ID', 'Image_Path']]
                print(f"Feature names: {feature_cols}")
        except Exception as e:
            print(f"Error in Phase B: {e}")
    else:
        print(f"CSV file not found: {DEFAULT_OUTPUT_FILE}")
        print("Run Phase A first to generate the CSV file.")
