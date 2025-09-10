"""
Enhanced Neural Network model for earthquake prediction.
Extends the existing implementation with advanced architecture and regularization.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import joblib
from typing import Dict, Tuple, Optional, Any, List
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config

class EnhancedNeuralNetworkModel:
    """Enhanced Neural Network model with advanced architecture and regularization."""
    
    def __init__(self, random_state: int = None):
        self.random_state = random_state or Config.RANDOM_STATE
        self.model = None
        self.is_fitted = False
        self.history = None
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds for reproducibility
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = None,
        batch_size: int = None,
        optimize_architecture: bool = False
    ) -> Dict[str, Any]:
        """
        Train the neural network model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            optimize_architecture: Whether to optimize architecture
            
        Returns:
            Training results dictionary
        """
        config = Config.get_model_config('neural_network')
        epochs = epochs or config.get('epochs', 100)
        batch_size = batch_size or config.get('batch_size', 32)
        
        # Build model
        if optimize_architecture:
            self.model = self._build_optimized_model(X_train.shape[1])
        else:
            self.model = self._build_model(X_train.shape[1])
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=20,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_fitted = True
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train, verbose=0)
        train_metrics = self._calculate_metrics(y_train, y_train_pred.ravel())
        
        results = {
            'model_type': 'NeuralNetwork',
            'train_metrics': train_metrics,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0],
            'epochs_trained': len(self.history.history['loss'])
        }
        
        if validation_data:
            y_val_pred = self.model.predict(X_val, verbose=0)
            val_metrics = self._calculate_metrics(y_val, y_val_pred.ravel())
            results['val_metrics'] = val_metrics
        
        # Cross-validation metrics
        cv_scores = self._cross_validate(X_train, y_train)
        results['cv_mean_r2'] = cv_scores.mean()
        results['cv_std_r2'] = cv_scores.std()
        
        self.logger.info(f"Neural Network trained: R² = {train_metrics['r2']:.4f}")
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X, verbose=0).ravel()
    
    def predict_with_uncertainty(
        self, 
        X: np.ndarray, 
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates using Monte Carlo Dropout.
        
        Args:
            X: Input features
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (predictions, uncertainty_estimates)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # For uncertainty estimation, we need a model with dropout layers
        # This implementation assumes dropout layers are present
        
        # Enable training mode to activate dropout during prediction
        predictions_mc = []
        
        for _ in range(n_samples):
            # Get prediction with dropout activated (training=True)
            pred = self.model(X, training=True).numpy().ravel()
            predictions_mc.append(pred)
        
        predictions_mc = np.array(predictions_mc)
        
        # Calculate mean and standard deviation
        predictions = predictions_mc.mean(axis=0)
        uncertainties = predictions_mc.std(axis=0)
        
        return predictions, uncertainties
    
    def _build_model(self, n_features: int) -> tf.keras.Model:
        """Build the neural network architecture."""
        config = Config.get_model_config('neural_network')
        hidden_layers = config.get('hidden_layers', [128, 64, 32])
        learning_rate = config.get('learning_rate', 0.001)
        
        model = Sequential()
        
        # Input layer
        model.add(Dense(
            hidden_layers[0], 
            activation='relu',
            input_shape=(n_features,),
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(
                units, 
                activation='relu',
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_optimized_model(self, n_features: int) -> tf.keras.Model:
        """Build optimized neural network with architecture search results."""
        
        model = Sequential()
        
        # Optimized architecture based on typical earthquake prediction tasks
        layers = [256, 128, 64, 32, 16]
        
        # Input layer
        model.add(Dense(
            layers[0],
            activation='relu',
            input_shape=(n_features,),
            kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        
        # Hidden layers with decreasing size and dropout
        for i, units in enumerate(layers[1:]):
            model.add(Dense(
                units,
                activation='relu',
                kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5)
            ))
            model.add(BatchNormalization())
            
            # Decreasing dropout rate
            dropout_rate = 0.4 - (i * 0.05)
            dropout_rate = max(dropout_rate, 0.1)
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile with optimized settings
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> np.ndarray:
        """Perform cross-validation to assess model performance with adaptive fold handling."""
        n_samples = len(X)
        
        # Adaptive cross-validation: adjust folds based on sample size
        # Ensure we have enough samples for each fold (at least 5 samples per fold)
        max_folds = min(cv_folds, max(2, n_samples // 5))
        
        if max_folds < 2:
            self.logger.warning(f"Insufficient data for cross-validation: {n_samples} samples. Using single score.")
            # Return a single score based on the full dataset (no cross-validation)
            temp_model = self._build_model(X.shape[1])
            temp_model.fit(X, y, epochs=10, batch_size=min(32, len(X)), verbose=0)
            y_pred = temp_model.predict(X, verbose=0)
            r2 = r2_score(y, y_pred.ravel())
            return np.array([r2])  # Return as array for consistency
        
        actual_folds = max_folds
        self.logger.info(f"Using {actual_folds} CV folds for {n_samples} samples (requested: {cv_folds})")
        
        kfold = KFold(n_splits=actual_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []
        
        for train_idx, val_idx in kfold.split(X):
            # Convert to numpy arrays to avoid pandas indexing issues
            X_array = X.values if hasattr(X, 'values') else X
            y_array = y.values if hasattr(y, 'values') else y
            
            X_fold_train, X_fold_val = X_array[train_idx], X_array[val_idx]
            y_fold_train, y_fold_val = y_array[train_idx], y_array[val_idx]
            
            # Build and train model for this fold
            fold_model = self._build_model(X_fold_train.shape[1])
            
            # Quick training for cross-validation
            fold_model.fit(
                X_fold_train, y_fold_train,
                epochs=50,  # Fewer epochs for CV
                batch_size=min(32, len(X_fold_train)),  # Adaptive batch size
                validation_data=(X_fold_val, y_fold_val),
                verbose=0,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
            )
            
            # Evaluate
            y_pred = fold_model.predict(X_fold_val, verbose=0)
            r2 = r2_score(y_fold_val, y_pred.ravel())
            cv_scores.append(r2)
        
        return np.array(cv_scores)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        if self.history is None:
            return {}
        return self.history.history
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        # Save Keras model
        model_path = filepath.replace('.pkl', '_keras_model.h5')
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'random_state': self.random_state,
            'model_path': model_path,
            'history': self.history.history if self.history else None
        }
        
        joblib.dump(metadata, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a saved model."""
        metadata = joblib.load(filepath)
        
        # Load Keras model
        model_path = metadata['model_path']
        self.model = tf.keras.models.load_model(model_path)
        
        self.random_state = metadata['random_state']
        self.is_fitted = True
        
        # Restore history if available
        if metadata.get('history'):
            class FakeHistory:
                def __init__(self, history_dict):
                    self.history = history_dict
            self.history = FakeHistory(metadata['history'])
        
        self.logger.info(f"Model loaded from {filepath}")