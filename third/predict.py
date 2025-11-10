import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class EmotionPredictor:
    """
    A comprehensive predictor class for facial emotion recognition.
    Supports multiple machine learning models including Neural Network (MLP), 
    Decision Trees, Naive Bayes, Random Forest, and Support Vector Machines.
    """
    
    def __init__(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        """
        Initialize the predictor with training, validation, and test data.
        
        Args:
            X_train: Training features (feature vectors from LBP+PCA or raw images)
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
        Prepare data for ML algorithms (flatten images if needed, or pass through feature vectors).
        
        Args:
            X: Input data (feature vectors or raw images)
            
        Returns:
            np.ndarray: Data ready for ML algorithms
        """
        if self.data_type == 'images':
            # Flatten images for ML
            return X.reshape(X.shape[0], -1)
        else:
            # Already feature vectors (e.g., from LBP+PCA)
            return X
    
    def train_neural_net(self, hidden_layer_sizes=(64, 32), alpha=1e-3, 
                        learning_rate_init=1e-3, max_iter=500, early_stopping=True,
                        validation_fraction=0.2, batch_size='auto', patience=10,
                        n_iter_per_check=20, min_improvement=0.0001):
        """
        Train a Neural Network (MLP) classifier for emotion recognition.
        
        Args:
            hidden_layer_sizes (tuple): Sizes of hidden layers (default: (64, 32))
            alpha (float): L2 regularization parameter (default: 1e-3)
            learning_rate_init (float): Initial learning rate (default: 1e-3)
            max_iter (int): Maximum number of iterations (default: 500)
            early_stopping (bool): Whether to use early stopping (default: True)
            validation_fraction (float): Fraction of training data for validation if X_val not provided (default: 0.2)
            batch_size (str or int): Size of minibatches (default: 'auto')
            patience (int): Number of checks without improvement before stopping (default: 10)
            n_iter_per_check (int): Number of iterations between validation checks (default: 20)
            min_improvement (float): Minimum improvement in validation accuracy to reset patience (default: 0.0001)
            
        Returns:
            dict or MLPClassifier: If using manual early stopping with X_val, returns dict with model and history.
                                   Otherwise returns trained model directly.
        """
        print("Training Neural Network (MLP)...")
        
        # Prepare data
        X_train_ml = self._prepare_data_for_traditional_ml(self.X_train)
        
        # Use manual early stopping if validation set is provided and early_stopping is enabled
        if self.X_val is not None and early_stopping:
            print("Using provided validation set for early stopping...")
            X_val_ml = self._prepare_data_for_traditional_ml(self.X_val)
            
            # Initialize model with warm_start for iterative training
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation='relu',
                solver='adam',
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=n_iter_per_check,
                warm_start=True,
                batch_size=batch_size,
                random_state=42,
                verbose=False
            )
            
            # Training loop with manual early stopping
            best_val_acc = 0.0
            best_model_state = None
            patience_counter = 0
            total_iterations = 0
            val_acc_history = []
            train_acc_history = []
            
            max_checks = max_iter // n_iter_per_check
            
            for check in range(max_checks):
                # Train for n_iter_per_check iterations
                model.fit(X_train_ml, self.y_train)
                total_iterations += n_iter_per_check
                
                # Evaluate on validation set
                y_val_pred = model.predict(X_val_ml)
                val_acc = accuracy_score(self.y_val, y_val_pred)
                val_acc_history.append(val_acc)
                
                # Track training accuracy
                y_train_pred = model.predict(X_train_ml)
                train_acc = accuracy_score(self.y_train, y_train_pred)
                train_acc_history.append(train_acc)
                
                print(f"Check {check+1}/{max_checks} (iter {total_iterations}): "
                      f"Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
                
                # Check if validation accuracy improved
                if val_acc > best_val_acc + min_improvement:
                    best_val_acc = val_acc
                    # Deep copy model state
                    import copy
                    best_model_state = copy.deepcopy(model)
                    patience_counter = 0
                    print(f"  → New best validation accuracy: {best_val_acc:.4f}")
                else:
                    patience_counter += 1
                    print(f"  → No improvement (patience: {patience_counter}/{patience})")
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {total_iterations} iterations")
                    break
            
            # Restore best model
            if best_model_state is not None:
                model = best_model_state
                print(f"Restored best model with validation accuracy: {best_val_acc:.4f}")
            
            # Store model
            self.models['neural_net'] = model
            
            print("Neural Network training completed")
            
            # Return model with training history
            return {
                'model': model,
                'val_acc_history': val_acc_history,
                'train_acc_history': train_acc_history,
                'total_iterations': total_iterations,
                'best_val_acc': best_val_acc,
                'early_stopped': patience_counter >= patience
            }
        
        else:
            # Fall back to standard training (original behavior)
            if self.X_val is None and early_stopping:
                print(f"Using internal validation_fraction={validation_fraction} for early stopping...")
            
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation='relu',
                solver='adam',
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                batch_size=batch_size,
                random_state=42,
                verbose=False
            )
            
            model.fit(X_train_ml, self.y_train)
            
            # Store model
            self.models['neural_net'] = model
            
            print("Neural Network training completed")
            return model
    
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
            model_type (str): Type of model to train ('neural_net', 'decision_tree', 'naive_bayes', 'random_forest', 'svm')
            **kwargs: Additional arguments for the specific training method
            
        Returns:
            Trained model or training history
        """
        if model_type == 'neural_net':
            return self.train_neural_net(**kwargs)
        elif model_type == 'decision_tree':
            return self.train_decision_tree(**kwargs)
        elif model_type == 'naive_bayes':
            return self.train_naive_bayes(**kwargs)
        elif model_type == 'random_forest':
            return self.train_random_forest(**kwargs)
        elif model_type == 'svm':
            return self.train_svm(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose from: neural_net, decision_tree, naive_bayes, random_forest, svm")
    
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
