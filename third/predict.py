import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. CNN training will be disabled.")


class EmotionPredictor:
    """
    A comprehensive predictor class for facial emotion recognition.
    Supports multiple machine learning models including CNN, Decision Trees,
    Naive Bayes, Random Forest, and Support Vector Machines.
    """
    
    def __init__(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        """
        Initialize the predictor with training, validation, and test data.
        
        Args:
            X_train: Training features (images or feature vectors)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            X_test: Test features (optional)
            y_test: Test labels (optional)
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_val = np.array(X_val) if X_val is not None else None
        self.y_val = np.array(y_val) if y_val is not None else None
        self.X_test = np.array(X_test) if X_test is not None else None
        self.y_test = np.array(y_test) if y_test is not None else None
        
        # Auto-detect data type
        self.data_type = self._detect_data_type()
        
        # Store trained models
        self.models = {}
        
        # Emotion mapping (same as processing.py)
        self.emotion_map = {
            'Anger': 0, 'Contempt': 1, 'Disgust': 2, 'Fear': 3, 
            'Happy': 4, 'Neutral': 5, 'Sad': 6, 'Surprised': 7
        }
        
        print(f"EmotionPredictor initialized with {self.data_type} data")
        print(f"Training set shape: {self.X_train.shape}")
        if self.X_val is not None:
            print(f"Validation set shape: {self.X_val.shape}")
        if self.X_test is not None:
            print(f"Test set shape: {self.X_test.shape}")
    
    def _detect_data_type(self):
        """
        Auto-detect if input data is raw images or feature vectors.
        
        Returns:
            str: 'images' or 'features'
        """
        # Check if data has 3+ dimensions (likely images)
        if len(self.X_train.shape) >= 3:
            return 'images'
        else:
            return 'features'
    
    def _prepare_data_for_traditional_ml(self, X):
        """
        Prepare data for traditional ML algorithms (flatten images if needed).
        
        Args:
            X: Input data
            
        Returns:
            np.ndarray: Flattened data ready for traditional ML
        """
        if self.data_type == 'images':
            # Flatten images for traditional ML
            return X.reshape(X.shape[0], -1)
        else:
            # Already feature vectors
            return X
    
    def _prepare_data_for_cnn(self, X):
        """
        Prepare data for CNN (reshape if needed).
        
        Args:
            X: Input data
            
        Returns:
            np.ndarray: Data ready for CNN
        """
        if self.data_type == 'images':
            # Ensure images have the right shape for CNN
            if len(X.shape) == 3:  # Grayscale images
                return X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
            else:
                return X
        else:
            raise ValueError("CNN requires image data, not feature vectors")
    
    def train_cnn(self, epochs=50, batch_size=32, learning_rate=0.001):
        """
        Train a CNN model for emotion recognition.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
            
        Returns:
            dict: Training history
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN training. Please install tensorflow.")
        
        if self.data_type != 'images':
            raise ValueError("CNN training requires image data, not feature vectors")
        
        print("Training CNN model...")
        
        # Prepare data for CNN
        X_train_cnn = self._prepare_data_for_cnn(self.X_train)
        X_val_cnn = self._prepare_data_for_cnn(self.X_val) if self.X_val is not None else None
        
        # Get image dimensions
        img_height, img_width = X_train_cnn.shape[1], X_train_cnn.shape[2]
        
        # Build CNN model
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(8, activation='softmax')  # 8 emotion classes
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train model
        if X_val_cnn is not None:
            history = model.fit(
                X_train_cnn, self.y_train,
                validation_data=(X_val_cnn, self.y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = model.fit(
                X_train_cnn, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        # Store model
        self.models['cnn'] = model
        
        print("CNN training completed")
        return history.history
    
    def train_decision_tree(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Train a Decision Tree classifier.
        
        Args:
            max_depth (int): Maximum depth of the tree
            min_samples_split (int): Minimum samples required to split
            min_samples_leaf (int): Minimum samples required at leaf nodes
            
        Returns:
            DecisionTreeClassifier: Trained model
        """
        print("Training Decision Tree...")
        
        # Prepare data
        X_train_ml = self._prepare_data_for_traditional_ml(self.X_train)
        
        # Create and train model
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        model.fit(X_train_ml, self.y_train)
        
        # Store model
        self.models['decision_tree'] = model
        
        print("Decision Tree training completed")
        return model
    
    def train_naive_bayes(self):
        """
        Train a Gaussian Naive Bayes classifier.
        
        Returns:
            GaussianNB: Trained model
        """
        print("Training Naive Bayes...")
        
        # Prepare data
        X_train_ml = self._prepare_data_for_traditional_ml(self.X_train)
        
        # Create and train model
        model = GaussianNB()
        model.fit(X_train_ml, self.y_train)
        
        # Store model
        self.models['naive_bayes'] = model
        
        print("Naive Bayes training completed")
        return model
    
    def train_random_forest(self, n_estimators=100, max_depth=None, min_samples_split=2):
        """
        Train a Random Forest classifier.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of trees
            min_samples_split (int): Minimum samples required to split
            
        Returns:
            RandomForestClassifier: Trained model
        """
        print("Training Random Forest...")
        
        # Prepare data
        X_train_ml = self._prepare_data_for_traditional_ml(self.X_train)
        
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_ml, self.y_train)
        
        # Store model
        self.models['random_forest'] = model
        
        print("Random Forest training completed")
        return model
    
    def train_svm(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        Train a Support Vector Machine classifier.
        
        Args:
            kernel (str): Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            C (float): Regularization parameter
            gamma (str or float): Kernel coefficient
            
        Returns:
            SVC: Trained model
        """
        print("Training SVM...")
        
        # Prepare data
        X_train_ml = self._prepare_data_for_traditional_ml(self.X_train)
        
        # Create and train model
        model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=42,
            probability=True
        )
        
        model.fit(X_train_ml, self.y_train)
        
        # Store model
        self.models['svm'] = model
        
        print("SVM training completed")
        return model
    
    def train(self, model_type, **kwargs):
        """
        Unified training interface that calls the appropriate training method.
        
        Args:
            model_type (str): Type of model to train ('cnn', 'decision_tree', 'naive_bayes', 'random_forest', 'svm')
            **kwargs: Additional arguments for the specific training method
            
        Returns:
            Trained model or training history
        """
        if model_type == 'cnn':
            return self.train_cnn(**kwargs)
        elif model_type == 'decision_tree':
            return self.train_decision_tree(**kwargs)
        elif model_type == 'naive_bayes':
            return self.train_naive_bayes(**kwargs)
        elif model_type == 'random_forest':
            return self.train_random_forest(**kwargs)
        elif model_type == 'svm':
            return self.train_svm(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose from: cnn, decision_tree, naive_bayes, random_forest, svm")
    
    def predict(self, model_type, X):
        """
        Make predictions using a trained model.
        
        Args:
            model_type (str): Type of model to use for prediction
            X: Input data for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        if model_type not in self.models:
            raise ValueError(f"Model '{model_type}' not found. Train the model first.")
        
        model = self.models[model_type]
        
        if model_type == 'cnn':
            X_pred = self._prepare_data_for_cnn(X)
            predictions = model.predict(X_pred)
            return np.argmax(predictions, axis=1)
        else:
            X_pred = self._prepare_data_for_traditional_ml(X)
            return model.predict(X_pred)
    
    def evaluate(self, model_type, dataset='test'):
        """
        Evaluate a trained model on validation or test set.
        
        Args:
            model_type (str): Type of model to evaluate
            dataset (str): Dataset to evaluate on ('val' or 'test')
            
        Returns:
            dict: Evaluation metrics
        """
        if model_type not in self.models:
            raise ValueError(f"Model '{model_type}' not found. Train the model first.")
        
        if dataset == 'val' and self.X_val is None:
            raise ValueError("Validation set not available")
        elif dataset == 'test' and self.X_test is None:
            raise ValueError("Test set not available")
        
        # Get data and labels
        if dataset == 'val':
            X_eval, y_eval = self.X_val, self.y_val
        else:
            X_eval, y_eval = self.X_test, self.y_test
        
        # Make predictions
        y_pred = self.predict(model_type, X_eval)
        
        # Calculate metrics
        accuracy = accuracy_score(y_eval, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': y_eval
        }
        
        return metrics
    
    def get_metrics(self, model_type, dataset='test'):
        """
        Get detailed metrics including confusion matrix and classification report.
        
        Args:
            model_type (str): Type of model to evaluate
            dataset (str): Dataset to evaluate on ('val' or 'test')
            
        Returns:
            dict: Detailed metrics including accuracy, confusion matrix, and classification report
        """
        metrics = self.evaluate(model_type, dataset)
        
        # Add confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(metrics['true_labels'], metrics['predictions'])
        
        # Add classification report
        emotion_names = list(self.emotion_map.keys())
        metrics['classification_report'] = classification_report(
            metrics['true_labels'], 
            metrics['predictions'], 
            target_names=emotion_names,
            output_dict=True
        )
        
        return metrics
    
    def get_available_models(self):
        """
        Get list of trained models.
        
        Returns:
            list: List of trained model names
        """
        return list(self.models.keys())
    
    def get_emotion_names(self):
        """
        Get emotion class names.
        
        Returns:
            list: List of emotion names
        """
        return list(self.emotion_map.keys())
