import os
import zipfile
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.feature import hog, local_binary_pattern
import random
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif
from mtcnn.mtcnn import MTCNN

try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. CNN features will be disabled.")


class EmotionImageProcessor:
    """
    A comprehensive processing class for facial emotion recognition dataset.
    Loads images, performs train-test splits, and provides feature extraction methods.
    
    """
    
    def __init__(self, dataset_path, train_subjects=None, val_subjects=None, test_subjects=None):
        """
        Initialize the processor by loading images and metadata.
        
        Args:
            dataset_path (str): Path to the facial-emotion-recognition dataset folder
            train_subjects (int, optional): Number of distinct persons to place in the training split.
            val_subjects (int, optional): Number of distinct persons to place in the validation split.
            test_subjects (int, optional): Number of distinct persons to place in the test split.
        """
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / "images"
        if not self.images_path.exists():
            self.images_path = self.dataset_path
        self.csv_path = self.dataset_path / "facial-emotion-recognition-dataset.csv"

        self.default_train_subjects = train_subjects
        self.default_val_subjects = val_subjects
        self.default_test_subjects = test_subjects
        
        # Emotion mapping
        self.emotion_map = {
            'Anger': 0, 'Contempt': 1, 'Disgust': 2, 'Fear': 3, 
            'Happy': 4, 'Neutral': 5, 'Sad': 6, 'Surprised': 7
        }
        
        # Initialize data storage
        self.images = []
        self.labels = []
        self.metadata = []
        self.image_shape = None
        
        # Train/validation/test split storage
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        # Load data
        self._load_images()
        self._load_metadata()
        
    def _load_images(self, target_size=(256, 256)):
        """Load all images from the dataset folders."""
        print("Loading images...")
        
        # Get all person folders (0-18)
        person_folders = sorted([f for f in os.listdir(self.images_path) 
                                if os.path.isdir(self.images_path / f) and f.isdigit()])
        
        for person_folder in person_folders:
            person_path = self.images_path / person_folder
            
            # Get all emotion images for this person
            emotion_files = [f for f in os.listdir(person_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for emotion_file in emotion_files:
                try:
                    # Extract emotion from filename
                    emotion_name = self._extract_emotion_from_filename(emotion_file)
                except ValueError as err:
                    print(f"Warning: {err}")
                    continue

                # Load image
                img_path = person_path / emotion_file
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                
                # Resize to target size to ensure consistent dimensions
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
                
                # Store image and label
                self.images.append(img_array)
                self.labels.append(self.emotion_map[emotion_name])
                self.metadata.append({
                    'person_id': int(person_folder),
                    'emotion': emotion_name,
                    'filename': emotion_file
                })
        
        # Set image shape
        self.image_shape = target_size
        
        # Convert to numpy arrays
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        print(f"Loaded {len(self.images)} images with shape {self.image_shape}")
        print(f"Emotion distribution: {np.bincount(self.labels)}")
        
    def _extract_emotion_from_filename(self, filename):
        """Extract emotion name from filename."""
        stem = Path(filename).stem

        if stem in self.emotion_map:
            return stem

        normalized = stem.capitalize()
        if normalized in self.emotion_map:
            return normalized

        raise ValueError(f"Unexpected emotion filename '{filename}'. "
                         "Expected one of: "
                         f"{', '.join(self.emotion_map.keys())}")

    def _resolve_image_path(self, person_id, filename):
        """Return absolute path to the stored image for a given metadata entry."""
        img_path = self.images_path / str(person_id) / filename
        if not img_path.exists():
            img_path = self.dataset_path / str(person_id) / filename
        return img_path

    def _crop_face(self, img, target_size, x_ratio=0.18, y_ratio=0.10,
                   width_ratio=0.64, height_ratio=0.75):
        """
        Crop a central face region from a square image and resize back to target size.
        """
        h, w = img.shape[:2]
        crop_w = int(w * width_ratio)
        crop_h = int(h * height_ratio)
        x1 = int(w * x_ratio)
        y1 = int(h * y_ratio)
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        cropped = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if cropped.size == 0:
            return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    def preprocess(self, target_size=256, min_conf=0.85, align=True, crop=True,
                   crop_params=None, grayscale=True, normalize=True,
                   clahe=False, contrast_min=False, detector=None,
                   save_dir=None):
        """
        Optionally run face detection and alignment followed by cropping, grayscale conversion,
        and normalization on the entire dataset. Updates internal image arrays in-place and
        saves the processed dataset to disk.
        """
        if not self.metadata:
            raise ValueError("Dataset not loaded. Initialize EmotionImageProcessor first.")

        detector = detector or MTCNN()
        crop_params = crop_params or {
            'x_ratio': 0.18,
            'y_ratio': 0.10,
            'width_ratio': 0.64,
            'height_ratio': 0.75
        }

        save_dir = Path(save_dir) if save_dir is not None else (self.dataset_path.parent / "facial-emotion-recognition-preprossesed")
        save_dir.mkdir(parents=True, exist_ok=True)

        processed_images = []
        processed_labels = []
        processed_metadata = []
        failed = []

        print(f"Preprocessing images with target_size={target_size}, min_conf={min_conf}")

        for meta in self.metadata:
            img_path = self._resolve_image_path(meta['person_id'], meta['filename'])
            if not img_path.exists():
                failed.append({'person_id': meta['person_id'], 'filename': meta['filename'], 'reason': 'file_missing'})
                print(f"Warning: File not found {img_path}")
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                failed.append({'person_id': meta['person_id'], 'filename': meta['filename'], 'reason': 'read_error'})
                print(f"Warning: Failed to read {img_path}")
                continue

            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

            aligned_img, success, confidence = self._align_single_face(
                img_gray, detector, target_size=target_size, min_conf=min_conf
            )

            if align:
                if not success:
                    failed.append({
                        'person_id': meta['person_id'],
                        'filename': meta['filename'],
                        'reason': f'align_failed(conf={confidence:.2f})'
                    })
                    continue
                processed = aligned_img
            else:
                processed = aligned_img if success else cv2.resize(
                    img_gray, (target_size, target_size), interpolation=cv2.INTER_LINEAR
                )

            if not align and not success:
                failed.append({
                    'person_id': meta['person_id'],
                    'filename': meta['filename'],
                    'reason': f'align_failed(conf={confidence:.2f})'
                })
                # Continue processing using resized grayscale image despite alignment failure.

            if crop:
                processed = self._crop_face(processed, target_size=target_size, **crop_params)

            if grayscale:
                if processed.ndim == 3:
                    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                if processed.ndim == 2:
                    processed = np.repeat(processed[..., np.newaxis], 3, axis=2)

            if clahe or contrast_min:
                img_uint8 = np.clip(processed * 255.0, 0, 255).astype(np.uint8)
                if clahe:
                    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    img_uint8 = clahe_obj.apply(img_uint8)
                if contrast_min:
                    blurred = cv2.GaussianBlur(img_uint8, (99, 99), 0)
                    blurred = np.where(blurred == 0, 1, blurred)
                    img_uint8 = cv2.divide(img_uint8, blurred, scale=128)
                processed = img_uint8.astype(np.float32) / 255.0 if normalize else img_uint8.astype(np.float32)
            else:
                processed = processed.astype(np.float32)
                if normalize:
                    processed = np.clip(processed, 0.0, 1.0)

            processed_images.append(processed)
            processed_labels.append(self.emotion_map[meta['emotion']])
            processed_metadata.append(meta)

            # Persist processed image to disk
            person_dir = save_dir / str(meta['person_id'])
            person_dir.mkdir(parents=True, exist_ok=True)
            out_stem = Path(meta['filename']).stem
            out_path = person_dir / f"{out_stem}.png"

            to_save = processed
            if to_save.dtype != np.uint8:
                max_val = float(np.max(to_save)) if to_save.size else 0.0
                if max_val <= 1.0:
                    to_save = np.clip(to_save * 255.0, 0, 255)
                else:
                    to_save = np.clip(to_save, 0, 255)

            if grayscale:
                cv2.imwrite(str(out_path), to_save.astype(np.uint8))
            else:
                if to_save.ndim == 2:
                    to_save = np.repeat(to_save[:, :, np.newaxis], 3, axis=2)
                cv2.imwrite(str(out_path), to_save.astype(np.uint8))

        if not processed_images:
            raise RuntimeError("Preprocessing failed for all images.")

        self.images = np.array(processed_images, dtype=np.float32)
        self.labels = np.array(processed_labels, dtype=np.int32)
        self.metadata = processed_metadata
        self.image_shape = (target_size, target_size) if grayscale else (target_size, target_size, 3)

        # Reset existing splits as data has changed
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

        print(f"Preprocessing completed: {len(processed_images)} images processed, "
              f"{len(failed)} images failed.")

        return {
            'processed_count': len(processed_images),
            'failed_count': len(failed),
            'failed': failed
        }
        
    def _load_metadata(self):
        """Load metadata from CSV file."""
        if self.csv_path.exists():
            self.metadata_df = pd.read_csv(self.csv_path, sep=';')
            print(f"Loaded metadata for {len(self.metadata_df)} persons")
        else:
            print("Warning: CSV metadata file not found")
            self.metadata_df = None
    
    def _target_landmarks(self, target_size):
        """
        Define canonical facial landmark positions for alignment.
        
        Args:
            target_size (int): Size of the target image (assumed square)
            
        Returns:
            np.ndarray: Array of 5 landmark points (left_eye, right_eye, nose, left_mouth, right_mouth)
        """
        W = H = target_size
        pts = np.array([
            (0.35 * W, 0.38 * H),  # left eye
            (0.65 * W, 0.38 * H),  # right eye
            (0.50 * W, 0.55 * H),  # nose tip
            (0.40 * W, 0.75 * H),  # left mouth
            (0.60 * W, 0.75 * H),  # right mouth
        ], dtype=np.float32)
        return pts
    
    def _align_single_face(self, img, detector, target_size=256, min_conf=0.85, border_mode=cv2.BORDER_REFLECT):
        """
        Align a single grayscale image using detected facial landmarks.
        
        Args:
            img (np.ndarray): Grayscale image as float32 array [0,1]
            detector: MTCNN detector instance
            target_size (int): Target size for aligned image
            min_conf (float): Minimum confidence for face detection
            border_mode: OpenCV border mode for warping
            
        Returns:
            tuple: (aligned_image, success, confidence)
        """
        # Convert grayscale to RGB for MTCNN detection
        img_uint8 = (img * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        
        # Detect faces
        detections = detector.detect_faces(img_rgb)
        
        # Pick largest face
        if not detections:
            return None, False, 0.0
        
        face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
        
        if face.get('confidence', 0.0) < min_conf:
            return None, False, face.get('confidence', 0.0)
        
        # Extract landmarks
        kps = face['keypoints']
        src = np.array([
            kps['left_eye'],
            kps['right_eye'],
            kps['nose'],
            kps['mouth_left'],
            kps['mouth_right'],
        ], dtype=np.float32)
        
        dst = self._target_landmarks(target_size)
        
        # Compute similarity transform
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            return None, False, face['confidence']
        
        # Apply transformation to original grayscale image
        aligned = cv2.warpAffine(
            img_uint8, M, (target_size, target_size),
            flags=cv2.INTER_LINEAR, borderMode=border_mode
        )
        
        # Convert back to float32 and normalize
        aligned = aligned.astype(np.float32) / 255.0
        
        return aligned, True, face['confidence']
    
    def _compute_split_counts(self, n_persons, train_subjects, val_subjects, test_subjects, val_size, test_size):
        use_explicit_counts = any(
            value is not None for value in (train_subjects, val_subjects, test_subjects)
        )

        if use_explicit_counts:
            if val_subjects is None or test_subjects is None:
                raise ValueError(
                    "Explicit subject splits require both val_subjects and test_subjects."
                )

            val_count = int(val_subjects)
            test_count = int(test_subjects)
            if val_count < 0 or test_count < 0:
                raise ValueError("Split sizes must be non-negative integers.")

            if train_subjects is None:
                train_count = n_persons - val_count - test_count
            else:
                train_count = int(train_subjects)

            if train_count < 0:
                raise ValueError("Requested split sizes exceed number of available persons.")

            total_requested = train_count + val_count + test_count
            if total_requested > n_persons:
                raise ValueError(
                    f"Requested {total_requested} persons but only {n_persons} are available."
                )

            # Add any remaining persons to the training split by default.
            if total_requested < n_persons:
                train_count += n_persons - total_requested

            return train_count, val_count, test_count

        if not 0 <= val_size < 1 or not 0 <= test_size < 1 or val_size + test_size >= 1:
            raise ValueError(
                "val_size and test_size must be fractions in [0, 1) whose sum is less than 1."
            )

        val_count = int(round(n_persons * val_size))
        test_count = int(round(n_persons * test_size))

        # Ensure we do not exhaust all persons with val/test rounding.
        if val_count + test_count >= n_persons:
            # Reserve at least one person for training if possible.
            overflow = (val_count + test_count) - (n_persons - 1)
            if overflow > 0:
                # Reduce the larger of the two counts first.
                if val_count >= test_count:
                    val_count = max(0, val_count - overflow)
                else:
                    test_count = max(0, test_count - overflow)

        train_count = n_persons - val_count - test_count

        return train_count, val_count, test_count

    def train_val_test_split(
        self,
        train_subjects=None,
        val_subjects=None,
        test_subjects=None,
        val_size=0.2,
        test_size=0.2,
        random_state=42,
        data=None,
        labels=None,
    ):
        """
        Split data into train, validation, and test sets by person (not by individual images).
        This ensures all emotions of one person stay together in either train, validation, or test.
        
        Args:
            train_subjects (int, optional): Number of persons to include in training split.
            val_subjects (int, optional): Number of persons to include in validation split.
            test_subjects (int, optional): Number of persons to include in test split.
            val_size (float): Proportion of persons to use for validation when explicit counts are not provided.
            test_size (float): Proportion of persons to use for testing when explicit counts are not provided.
            random_state (int): Random seed for reproducibility
            data (np.ndarray, optional): Feature matrix aligned with internal metadata order.
            labels (np.ndarray, optional): Labels aligned with internal metadata order.

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if self.metadata is None or not self.metadata:
            raise ValueError("No metadata available to perform a train/validation/test split.")

        feature_source = np.asarray(data if data is not None else self.images)
        label_source = np.asarray(labels if labels is not None else self.labels)

        if feature_source.shape[0] != len(self.metadata):
            raise ValueError(
                "Feature array length does not match metadata length. Ensure data is aligned."
            )

        if label_source.shape[0] != len(self.metadata):
            raise ValueError(
                "Label array length does not match metadata length. Ensure labels are aligned."
            )

        train_subjects = (
            train_subjects if train_subjects is not None else self.default_train_subjects
        )
        val_subjects = (
            val_subjects if val_subjects is not None else self.default_val_subjects
        )
        test_subjects = (
            test_subjects if test_subjects is not None else self.default_test_subjects
        )

        # Get unique person IDs
        person_ids = sorted({meta['person_id'] for meta in self.metadata})
        
        # Set random seed
        rng = random.Random(random_state)
        rng.shuffle(person_ids)

        n_persons = len(person_ids)
        train_count, val_count, test_count = self._compute_split_counts(
            n_persons,
            train_subjects,
            val_subjects,
            test_subjects,
            val_size,
            test_size,
        )

        train_persons = person_ids[:train_count]
        val_persons = person_ids[train_count:train_count + val_count]
        test_persons = person_ids[train_count + val_count:train_count + val_count + test_count]
        
        # Create masks for train/val/test
        train_mask = np.array([meta['person_id'] in train_persons for meta in self.metadata])
        val_mask = np.array([meta['person_id'] in val_persons for meta in self.metadata])
        test_mask = np.array([meta['person_id'] in test_persons for meta in self.metadata])
        
        # Split data
        self.X_train = feature_source[train_mask]
        self.y_train = label_source[train_mask]
        self.X_val = feature_source[val_mask]
        self.y_val = label_source[val_mask]
        self.X_test = feature_source[test_mask]
        self.y_test = label_source[test_mask]

        if self.X_train.size:
            self.image_shape = self.X_train.shape[1:]
        
        print(f"Train set: {len(self.X_train)} images from {len(train_persons)} persons")
        print(f"Validation set: {len(self.X_val)} images from {len(val_persons)} persons")
        print(f"Test set: {len(self.X_test)} images from {len(test_persons)} persons")
        print(
            f"Train emotion distribution: {np.bincount(self.y_train) if len(self.y_train) else '[]'}"
        )
        print(
            f"Validation emotion distribution: {np.bincount(self.y_val) if len(self.y_val) else '[]'}"
        )
        print(
            f"Test emotion distribution: {np.bincount(self.y_test) if len(self.y_test) else '[]'}"
        )

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    def _compute_lbp_matrix(self, images, n_points=24, radius=3, method='uniform', flatten=True):
        """
        Compute Local Binary Pattern representations for a collection of images.

        Args:
            images (Iterable[np.ndarray]): Images to transform.
            n_points (int): Number of circularly symmetric neighbor points.
            radius (int): Radius of circle.
            method (str): LBP method.
            flatten (bool): Whether to flatten each LBP image to a 1D vector.

        Returns:
            np.ndarray: Matrix of LBP representations.
        """
        lbp_features = []

        for img in images:
            if img.ndim == 3 and img.shape[-1] == 3:
                img_gray = cv2.cvtColor(
                    (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8),
                    cv2.COLOR_BGR2GRAY,
                )
            else:
                if img.max() <= 1.0:
                    img_gray = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
                else:
                    img_gray = np.clip(img, 0, 255).astype(np.uint8)

            lbp = local_binary_pattern(img_gray, n_points, radius, method=method).astype(np.float32)
            lbp_features.append(lbp.ravel() if flatten else lbp)

        if not lbp_features:
            raise ValueError("No images available for LBP computation.")

        return np.stack(lbp_features)

    def preprocess_and_split_with_lbp(
        self,
        train_subjects,
        val_subjects,
        test_subjects,
        preprocess_kwargs=None,
        lbp_kwargs=None,
        random_state=42,
    ):
        """
        Run preprocessing followed by LBP feature extraction and perform a subject-level split.

        Args:
            train_subjects (int): Number of persons for the training split.
            val_subjects (int): Number of persons for the validation split.
            test_subjects (int): Number of persons for the test split.
            preprocess_kwargs (dict, optional): Keyword arguments forwarded to preprocess().
            lbp_kwargs (dict, optional): Keyword arguments forwarded to the LBP extraction.
            random_state (int): Deterministic seed for the subject shuffle.

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        preprocess_kwargs = preprocess_kwargs or {}
        lbp_kwargs = lbp_kwargs or {}

        # Step 1: preprocess raw images.
        self.preprocess(**preprocess_kwargs)

        # Step 2: compute LBP representations and flatten them.
        lbp_matrix = self._compute_lbp_matrix(self.images, **lbp_kwargs)

        # Step 3: perform subject-level split on the transformed data.
        return self.train_val_test_split(
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
            random_state=random_state,
            data=lbp_matrix,
            labels=self.labels,
        )
    
    def apply_alignment(self, target_size=256, min_conf=0.85, border_mode=cv2.BORDER_REFLECT):
        """
        Apply face alignment to train, validation, and test images using MTCNN.
        Skips images where face detection fails and removes them from the dataset.
        
        Args:
            target_size (int): Target size for aligned images
            min_conf (float): Minimum confidence for face detection
            border_mode: OpenCV border mode for warping
            
        Returns:
            dict: Statistics about alignment results
        """

        if self.X_train is None or self.X_val is None or self.X_test is None:
            raise ValueError("Must call train_val_test_split() first")
        
        print(f"Applying face alignment with target_size={target_size}, min_conf={min_conf}")
        
        # Initialize MTCNN detector
        detector = MTCNN()
        
        # Get person IDs from train/val/test split (recreate the split logic)
        person_ids = list(set([meta['person_id'] for meta in self.metadata]))
        random.seed(42)  # Use same seed as train_val_test_split
        random.shuffle(person_ids)
        
        # Calculate split indices (same as train_val_test_split default)
        n_persons = len(person_ids)
        val_split_idx = int(n_persons * 0.6)  # 60% train
        test_split_idx = int(n_persons * 0.8)  # 80% train+val
        
        train_person_ids = person_ids[:val_split_idx]
        val_person_ids = person_ids[val_split_idx:test_split_idx]
        test_person_ids = person_ids[test_split_idx:]
        
        # Create masks for train/val/test metadata
        train_mask = np.array([meta['person_id'] in train_person_ids for meta in self.metadata])
        val_mask = np.array([meta['person_id'] in val_person_ids for meta in self.metadata])
        test_mask = np.array([meta['person_id'] in test_person_ids for meta in self.metadata])
        
        train_metadata = [meta for meta, mask in zip(self.metadata, train_mask) if mask]
        val_metadata = [meta for meta, mask in zip(self.metadata, val_mask) if mask]
        test_metadata = [meta for meta, mask in zip(self.metadata, test_mask) if mask]
        
        # Align training images
        aligned_train_images = []
        aligned_train_labels = []
        train_failed = []
        
        for i, img in enumerate(self.X_train):
            aligned_img, success, confidence = self._align_single_face(
                img, detector, target_size, min_conf, border_mode
            )
            
            if success:
                aligned_train_images.append(aligned_img)
                aligned_train_labels.append(self.y_train[i])
            else:
                # Find corresponding metadata
                if i < len(train_metadata):
                    meta = train_metadata[i]
                    train_failed.append({
                        'person_id': meta['person_id'],
                        'emotion': meta['emotion'],
                        'confidence': confidence
                    })
                    print(f"Warning: Failed to align train image - Person {meta['person_id']}, "
                          f"Emotion: {meta['emotion']}, Confidence: {confidence:.3f}")
        
        # Align validation images
        aligned_val_images = []
        aligned_val_labels = []
        val_failed = []
        
        for i, img in enumerate(self.X_val):
            aligned_img, success, confidence = self._align_single_face(
                img, detector, target_size, min_conf, border_mode
            )
            
            if success:
                aligned_val_images.append(aligned_img)
                aligned_val_labels.append(self.y_val[i])
            else:
                # Find corresponding metadata
                if i < len(val_metadata):
                    meta = val_metadata[i]
                    val_failed.append({
                        'person_id': meta['person_id'],
                        'emotion': meta['emotion'],
                        'confidence': confidence
                    })
                    print(f"Warning: Failed to align validation image - Person {meta['person_id']}, "
                          f"Emotion: {meta['emotion']}, Confidence: {confidence:.3f}")
        
        # Align test images
        aligned_test_images = []
        aligned_test_labels = []
        test_failed = []
        
        for i, img in enumerate(self.X_test):
            aligned_img, success, confidence = self._align_single_face(
                img, detector, target_size, min_conf, border_mode
            )
            
            if success:
                aligned_test_images.append(aligned_img)
                aligned_test_labels.append(self.y_test[i])
            else:
                # Find corresponding metadata
                if i < len(test_metadata):
                    meta = test_metadata[i]
                    test_failed.append({
                        'person_id': meta['person_id'],
                        'emotion': meta['emotion'],
                        'confidence': confidence
                    })
                    print(f"Warning: Failed to align test image - Person {meta['person_id']}, "
                          f"Emotion: {meta['emotion']}, Confidence: {confidence:.3f}")
        
        # Update internal arrays
        self.X_train = np.array(aligned_train_images)
        self.y_train = np.array(aligned_train_labels)
        self.X_val = np.array(aligned_val_images)
        self.y_val = np.array(aligned_val_labels)
        self.X_test = np.array(aligned_test_images)
        self.y_test = np.array(aligned_test_labels)
        
        # Update image shape
        self.image_shape = (target_size, target_size)
        
        # Prepare statistics
        stats = {
            'original_train_count': len(self.X_train) + len(train_failed),
            'aligned_train_count': len(self.X_train),
            'train_failed_count': len(train_failed),
            'original_val_count': len(self.X_val) + len(val_failed),
            'aligned_val_count': len(self.X_val),
            'val_failed_count': len(val_failed),
            'original_test_count': len(self.X_test) + len(test_failed),
            'aligned_test_count': len(self.X_test),
            'test_failed_count': len(test_failed),
            'train_failed': train_failed,
            'val_failed': val_failed,
            'test_failed': test_failed
        }
        
        print(f"Face alignment completed:")
        print(f"  Training: {stats['aligned_train_count']}/{stats['original_train_count']} images aligned")
        print(f"  Validation: {stats['aligned_val_count']}/{stats['original_val_count']} images aligned")
        print(f"  Testing: {stats['aligned_test_count']}/{stats['original_test_count']} images aligned")
        print(f"  Total failed: {stats['train_failed_count'] + stats['val_failed_count'] + stats['test_failed_count']} images")
        
        return stats
    
    
    def apply_lbp(self, n_points=24, radius=3, method='uniform', n_bins=256):
        """
        Extract Local Binary Patterns (LBP) features from current images.
        
        Args:
            n_points (int): Number of circularly symmetric neighbor set points
            radius (int): Radius of circle
            method (str): Method to determine the pattern ('uniform', 'nri_uniform', 'var')
            n_bins (int): Number of bins for histogram
            
        Returns:
            tuple: (lbp_train_features, lbp_val_features, lbp_test_features)
        """
        if self.X_train is None or self.X_val is None or self.X_test is None:
            raise ValueError("Must call train_val_test_split() first")
        
        print(f"Extracting LBP features with n_points={n_points}, radius={radius}, method={method}")
        
        # Extract LBP features from training images
        lbp_train_features = []
        for img in self.X_train:
            # Convert to uint8 for LBP
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Compute LBP
            lbp = local_binary_pattern(img_uint8, n_points, radius, method=method)
            
            # Compute histogram
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            
            # Normalize histogram
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-7)  # Avoid division by zero
            
            lbp_train_features.append(hist)
        
        # Extract LBP features from validation images
        lbp_val_features = []
        for img in self.X_val:
            # Convert to uint8 for LBP
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Compute LBP
            lbp = local_binary_pattern(img_uint8, n_points, radius, method=method)
            
            # Compute histogram
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            
            # Normalize histogram
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-7)  # Avoid division by zero
            
            lbp_val_features.append(hist)
        
        # Extract LBP features from test images
        lbp_test_features = []
        for img in self.X_test:
            # Convert to uint8 for LBP
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Compute LBP
            lbp = local_binary_pattern(img_uint8, n_points, radius, method=method)
            
            # Compute histogram
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            
            # Normalize histogram
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-7)  # Avoid division by zero
            
            lbp_test_features.append(hist)
        
        lbp_train_features = np.array(lbp_train_features)
        lbp_val_features = np.array(lbp_val_features)
        lbp_test_features = np.array(lbp_test_features)
        
        print(f"LBP features extracted: train shape {lbp_train_features.shape}, "
              f"val shape {lbp_val_features.shape}, test shape {lbp_test_features.shape}")
        
        return lbp_train_features, lbp_val_features, lbp_test_features

    # Getter methods
    def get_train_images(self):
        """Return current training images."""
        return self.X_train
    
    def get_test_images(self):
        """Return current test images."""
        return self.X_test
    
    def get_val_images(self):
        """Return current validation images."""
        return self.X_val
    
    def get_train_labels(self):
        """Return training labels."""
        return self.y_train
    
    def get_val_labels(self):
        """Return validation labels."""
        return self.y_val
    
    def get_test_labels(self):
        """Return test labels."""
        return self.y_test
    
    def get_image_shape(self):
        """Return image shape (height, width)."""
        return self.image_shape
    
    def get_emotion_map(self):
        """Return emotion mapping dictionary."""
        return self.emotion_map.copy()
