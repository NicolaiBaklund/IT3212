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
    
    def __init__(self, dataset_path):
        """
        Initialize the processor by loading images and metadata.
        
        Args:
            dataset_path (str): Path to the facial-emotion-recognition dataset folder
        """
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / "images"
        self.csv_path = self.dataset_path / "facial-emotion-recognition-dataset.csv"
        
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
                # Extract emotion from filename
                emotion_name = self._extract_emotion_from_filename(emotion_file)
                if emotion_name in self.emotion_map:
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
        filename_lower = filename.lower()
        for emotion in self.emotion_map.keys():
            if emotion.lower() in filename_lower:
                return emotion
        return None
        
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
    
    def train_val_test_split(self, val_size=0.2, test_size=0.2, random_state=42):
        """
        Split data into train, validation, and test sets by person (not by individual images).
        This ensures all emotions of one person stay together in either train, validation, or test.
        
        Args:
            val_size (float): Proportion of persons to use for validation
            test_size (float): Proportion of persons to use for testing
            random_state (int): Random seed for reproducibility
        """
        # Get unique person IDs
        person_ids = list(set([meta['person_id'] for meta in self.metadata]))
        
        # Set random seed
        random.seed(random_state)
        random.shuffle(person_ids)
        
        # Calculate split indices
        n_persons = len(person_ids)
        val_split_idx = int(n_persons * (1 - val_size - test_size))
        test_split_idx = int(n_persons * (1 - test_size))
        
        # Split persons
        train_persons = person_ids[:val_split_idx]
        val_persons = person_ids[val_split_idx:test_split_idx]
        test_persons = person_ids[test_split_idx:]
        
        # Create masks for train/val/test
        train_mask = np.array([meta['person_id'] in train_persons for meta in self.metadata])
        val_mask = np.array([meta['person_id'] in val_persons for meta in self.metadata])
        test_mask = np.array([meta['person_id'] in test_persons for meta in self.metadata])
        
        # Split data
        self.X_train = self.images[train_mask]
        self.y_train = self.labels[train_mask]
        self.X_val = self.images[val_mask]
        self.y_val = self.labels[val_mask]
        self.X_test = self.images[test_mask]
        self.y_test = self.labels[test_mask]
        
        print(f"Train set: {len(self.X_train)} images from {len(train_persons)} persons")
        print(f"Validation set: {len(self.X_val)} images from {len(val_persons)} persons")
        print(f"Test set: {len(self.X_test)} images from {len(test_persons)} persons")
        print(f"Train emotion distribution: {np.bincount(self.y_train)}")
        print(f"Validation emotion distribution: {np.bincount(self.y_val)}")
        print(f"Test emotion distribution: {np.bincount(self.y_test)}")
    
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
    
    def apply_log(self, sigma=1.0, ksize=5):
        """
        Apply Laplacian of Gaussian (LoG) filter to train and test images.
        Updates the internal image arrays with filtered versions.
        
        Args:
            sigma (float): Standard deviation for Gaussian kernel
            ksize (int): Kernel size for Gaussian blur
        """
        if self.X_train is None or self.X_val is None or self.X_test is None:
            raise ValueError("Must call train_val_test_split() first")
        
        print(f"Applying LoG filter with sigma={sigma}, ksize={ksize}")
        
        # Apply LoG to training images
        filtered_train = []
        for img in self.X_train:
            # Convert to uint8 for OpenCV
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img_uint8, (ksize, ksize), sigma)
            
            # Apply Laplacian
            log_filtered = cv2.Laplacian(blurred, cv2.CV_64F)
            
            # Convert back to float32 and normalize
            log_filtered = np.abs(log_filtered).astype(np.float32) / 255.0
            filtered_train.append(log_filtered)
        
        # Apply LoG to validation images
        filtered_val = []
        for img in self.X_val:
            # Convert to uint8 for OpenCV
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img_uint8, (ksize, ksize), sigma)
            
            # Apply Laplacian
            log_filtered = cv2.Laplacian(blurred, cv2.CV_64F)
            
            # Convert back to float32 and normalize
            log_filtered = np.abs(log_filtered).astype(np.float32) / 255.0
            filtered_val.append(log_filtered)
        
        # Apply LoG to test images
        filtered_test = []
        for img in self.X_test:
            # Convert to uint8 for OpenCV
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img_uint8, (ksize, ksize), sigma)
            
            # Apply Laplacian
            log_filtered = cv2.Laplacian(blurred, cv2.CV_64F)
            
            # Convert back to float32 and normalize
            log_filtered = np.abs(log_filtered).astype(np.float32) / 255.0
            filtered_test.append(log_filtered)
        
        # Update internal arrays
        self.X_train = np.array(filtered_train)
        self.X_val = np.array(filtered_val)
        self.X_test = np.array(filtered_test)
        
        print("LoG filtering completed")
    
    def apply_hog(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """
        Extract Histogram of Oriented Gradients (HoG) features from current images.
        
        Args:
            orientations (int): Number of orientation bins
            pixels_per_cell (tuple): Size of a cell in pixels
            cells_per_block (tuple): Number of cells in each block
            
        Returns:
            tuple: (hog_train_features, hog_val_features, hog_test_features)
        """
        if self.X_train is None or self.X_val is None or self.X_test is None:
            raise ValueError("Must call train_val_test_split() first")
        
        print(f"Extracting HoG features with orientations={orientations}, "
              f"pixels_per_cell={pixels_per_cell}, cells_per_block={cells_per_block}")
        
        # Extract HoG features from training images
        hog_train_features = []
        for img in self.X_train:
            # Convert to uint8 for HoG
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Extract HoG features
            hog_features = hog(img_uint8, 
                              orientations=orientations,
                              pixels_per_cell=pixels_per_cell,
                              cells_per_block=cells_per_block,
                              feature_vector=True)
            hog_train_features.append(hog_features)
        
        # Extract HoG features from validation images
        hog_val_features = []
        for img in self.X_val:
            # Convert to uint8 for HoG
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Extract HoG features
            hog_features = hog(img_uint8,
                              orientations=orientations,
                              pixels_per_cell=pixels_per_cell,
                              cells_per_block=cells_per_block,
                              feature_vector=True)
            hog_val_features.append(hog_features)
        
        # Extract HoG features from test images
        hog_test_features = []
        for img in self.X_test:
            # Convert to uint8 for HoG
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Extract HoG features
            hog_features = hog(img_uint8,
                              orientations=orientations,
                              pixels_per_cell=pixels_per_cell,
                              cells_per_block=cells_per_block,
                              feature_vector=True)
            hog_test_features.append(hog_features)
        
        hog_train_features = np.array(hog_train_features)
        hog_val_features = np.array(hog_val_features)
        hog_test_features = np.array(hog_test_features)
        
        print(f"HoG features extracted: train shape {hog_train_features.shape}, "
              f"val shape {hog_val_features.shape}, test shape {hog_test_features.shape}")
        
        return hog_train_features, hog_val_features, hog_test_features
    
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
    
    def apply_cnn_features(self, model_name='resnet50', layer='avg_pool', input_size=(224, 224)):
        """
        Extract deep learning features using pre-trained CNN models.
        
        Args:
            model_name (str): Name of pre-trained model ('resnet50', 'vgg16', 'mobilenet_v2')
            layer (str): Layer name to extract features from
            input_size (tuple): Input image size for the model
            
        Returns:
            tuple: (cnn_train_features, cnn_val_features, cnn_test_features)
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN feature extraction. Please install tensorflow.")
        
        if self.X_train is None or self.X_val is None or self.X_test is None:
            raise ValueError("Must call train_val_test_split() first")
        
        print(f"Extracting CNN features using {model_name} from layer {layer}")
        
        # Load pre-trained model
        if model_name.lower() == 'resnet50':
            model = ResNet50(weights='imagenet', include_top=False, input_shape=(*input_size, 3))
            preprocess_func = resnet_preprocess
        elif model_name.lower() == 'vgg16':
            model = VGG16(weights='imagenet', include_top=False, input_shape=(*input_size, 3))
            preprocess_func = vgg_preprocess
        elif model_name.lower() == 'mobilenet_v2':
            model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*input_size, 3))
            preprocess_func = mobilenet_preprocess
        else:
            raise ValueError(f"Unsupported model: {model_name}. Choose from 'resnet50', 'vgg16', 'mobilenet_v2'")
        
        # Create feature extraction model
        feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer).output)
        
        def extract_features_from_images(images):
            """Extract features from a batch of images."""
            # Convert grayscale to RGB
            rgb_images = []
            for img in images:
                # Resize to model input size
                img_resized = cv2.resize(img, input_size)
                # Convert grayscale to RGB
                img_rgb = cv2.cvtColor((img_resized * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                rgb_images.append(img_rgb)
            
            rgb_images = np.array(rgb_images)
            
            # Preprocess for the specific model
            rgb_images = preprocess_func(rgb_images)
            
            # Extract features
            features = feature_extractor.predict(rgb_images, verbose=0)
            
            # Flatten features
            features_flat = features.reshape(features.shape[0], -1)
            
            return features_flat
        
        # Extract features from training images
        print("Extracting features from training images...")
        cnn_train_features = extract_features_from_images(self.X_train)
        
        # Extract features from validation images
        print("Extracting features from validation images...")
        cnn_val_features = extract_features_from_images(self.X_val)
        
        # Extract features from test images
        print("Extracting features from test images...")
        cnn_test_features = extract_features_from_images(self.X_test)
        
        print(f"CNN features extracted: train shape {cnn_train_features.shape}, "
              f"val shape {cnn_val_features.shape}, test shape {cnn_test_features.shape}")
        
        return cnn_train_features, cnn_val_features, cnn_test_features
    
    def apply_pca(self, n_components=50):
        """
        Apply Principal Component Analysis (PCA) to current images.
        Fits PCA on training data and transforms train, validation, and test data.
        
        Args:
            n_components (int): Number of principal components to retain
            
        Returns:
            tuple: (pca_train_features, pca_val_features, pca_test_features, pca_object)
        """
        if self.X_train is None or self.X_val is None or self.X_test is None:
            raise ValueError("Must call train_val_test_split() first")
        
        print(f"Applying PCA with {n_components} components")
        
        # Flatten images for PCA
        X_train_flat = self.X_train.reshape(self.X_train.shape[0], -1)
        X_val_flat = self.X_val.reshape(self.X_val.shape[0], -1)
        X_test_flat = self.X_test.reshape(self.X_test.shape[0], -1)
        
        # Compute mean from training data
        mean_image = np.mean(X_train_flat, axis=0)
        
        # Center the data
        X_train_centered = X_train_flat - mean_image
        X_val_centered = X_val_flat - mean_image
        X_test_centered = X_test_flat - mean_image
        
        # Compute covariance matrix
        n_samples = X_train_centered.shape[0]
        cov_matrix = (X_train_centered @ X_train_centered.T) / (n_samples - 1)
        
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        
        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Keep only positive eigenvalues
        mask = eigvals > 1e-12
        eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]
        
        # Map back to feature space
        components = []
        for i in range(min(n_components, len(eigvals))):
            v = X_train_centered.T @ eigvecs[:, i]
            v /= np.sqrt((n_samples - 1) * eigvals[i])
            components.append(v)
        
        components = np.column_stack(components)
        
        # Transform data
        pca_train_features = X_train_centered @ components
        pca_val_features = X_val_centered @ components
        pca_test_features = X_test_centered @ components
        
        # Create PCA object for later use
        pca_object = {
            'components': components,
            'mean': mean_image,
            'explained_variance': eigvals[:n_components],
            'n_components': n_components
        }
        
        print(f"PCA completed: train shape {pca_train_features.shape}, "
              f"val shape {pca_val_features.shape}, test shape {pca_test_features.shape}")
        
        return pca_train_features, pca_val_features, pca_test_features, pca_object
    
    def apply_select_k_best(self, k=100, score_func=f_classif):
        """
        Apply SelectKBest feature selection to reduce dimensionality.
        Works on any feature matrix (HoG, LBP, CNN features).
        
        Args:
            k (int): Number of top features to select
            score_func: Scoring function for feature selection (default: f_classif)
            
        Returns:
            tuple: (selected_train_features, selected_val_features, selected_test_features, selector_object)
        """
        if self.X_train is None or self.X_val is None or self.X_test is None:
            raise ValueError("Must call train_val_test_split() first")
        
        print(f"Applying SelectKBest with k={k} features")
        
        # Flatten images for feature selection
        X_train_flat = self.X_train.reshape(self.X_train.shape[0], -1)
        X_val_flat = self.X_val.reshape(self.X_val.shape[0], -1)
        X_test_flat = self.X_test.reshape(self.X_test.shape[0], -1)
        
        # Initialize SelectKBest
        selector = SelectKBest(score_func=score_func, k=k)
        
        # Fit on training data
        X_train_selected = selector.fit_transform(X_train_flat, self.y_train)
        
        # Transform validation and test data
        X_val_selected = selector.transform(X_val_flat)
        X_test_selected = selector.transform(X_test_flat)
        
        print(f"SelectKBest completed: train shape {X_train_selected.shape}, "
              f"val shape {X_val_selected.shape}, test shape {X_test_selected.shape}")
        
        # Create selector object for later use
        selector_object = {
            'selector': selector,
            'selected_features': selector.get_support(),
            'feature_scores': selector.scores_,
            'k': k
        }
        
        return X_train_selected, X_val_selected, X_test_selected, selector_object
    
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
