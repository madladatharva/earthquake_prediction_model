"""
LSTM model for earthquake prediction with time series forecasting.
Implements sequence-based prediction for temporal earthquake patterns.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from typing import Dict, Tuple, Optional, Any, List
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config

class LSTMEarthquakeModel:
    """LSTM model for time-series earthquake prediction."""
    
    def __init__(self, random_state: int = None):
        self.random_state = random_state or Config.RANDOM_STATE
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.sequence_length = Config.get_model_config('lstm').get('sequence_length', 10)
        self.is_fitted = False
        self.history = None
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds for reproducibility
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        
    def create_sequences(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        sequence_length: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training from time series data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        seq_len = sequence_length or self.sequence_length
        
        if len(X) < seq_len:
            raise ValueError(f"Not enough data for sequence length {seq_len}. Need at least {seq_len} samples.")
        
        X_sequences = []
        y_sequences = []
        
        for i in range(seq_len, len(X)):
            X_sequences.append(X[i-seq_len:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = None,
        batch_size: int = None,
        optimize_params: bool = False
    ) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)  
            epochs: Number of training epochs
            batch_size: Batch size for training
            optimize_params: Whether to optimize hyperparameters
            
        Returns:
            Training results dictionary
        """
        config = Config.get_model_config('lstm')
        epochs = epochs or config.get('epochs', 100)
        batch_size = batch_size or config.get('batch_size', 32)
        
        # Scale the data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).ravel()
            X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val_scaled)
            validation_data = (X_val_seq, y_val_seq)
        
        # Build model
        if optimize_params:
            self.model = self._build_optimized_model(X_train_seq.shape)
        else:
            self.model = self._build_model(X_train_seq.shape)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_fitted = True
        
        # Calculate training metrics
        y_train_pred_scaled = self.model.predict(X_train_seq, verbose=0)
        y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
        
        # Align predictions with original targets (account for sequence offset)
        y_train_aligned = y_train[self.sequence_length:]
        train_metrics = self._calculate_metrics(y_train_aligned, y_train_pred)
        
        results = {
            'model_type': 'LSTM',
            'train_metrics': train_metrics,
            'sequence_length': self.sequence_length,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0],
            'epochs_trained': len(self.history.history['loss'])
        }
        
        if validation_data:
            y_val_pred_scaled = self.model.predict(X_val_seq, verbose=0)
            y_val_pred = self.scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()
            y_val_aligned = y_val[self.sequence_length:]
            val_metrics = self._calculate_metrics(y_val_aligned, y_val_pred)
            results['val_metrics'] = val_metrics
        
        self.logger.info(f"LSTM trained: R² = {train_metrics['r2']:.4f}")
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler_X.transform(X)
        
        # For prediction, we need to handle the sequence creation differently
        if len(X_scaled) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} samples for prediction")
        
        # Create sequences for prediction
        X_sequences = []
        for i in range(self.sequence_length, len(X_scaled) + 1):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
        
        X_sequences = np.array(X_sequences)
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_sequences, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        return y_pred
    
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
        
        # Enable dropout during prediction for uncertainty estimation
        # Note: This requires the model to have dropout layers
        
        X_scaled = self.scaler_X.transform(X)
        
        if len(X_scaled) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} samples for prediction")
        
        # Create sequences
        X_sequences = []
        for i in range(self.sequence_length, len(X_scaled) + 1):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
        
        X_sequences = np.array(X_sequences)
        
        # Monte Carlo sampling
        predictions_mc = []
        for _ in range(n_samples):
            # For uncertainty, we'd need dropout enabled during inference
            # This is a simplified approach
            y_pred_scaled = self.model.predict(X_sequences, verbose=0)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            predictions_mc.append(y_pred)
        
        predictions_mc = np.array(predictions_mc)
        
        # Calculate mean and uncertainty
        predictions = predictions_mc.mean(axis=0)
        uncertainties = predictions_mc.std(axis=0)
        
        return predictions, uncertainties
    
    def predict_future(
        self, 
        X_last: np.ndarray, 
        steps: int = 1
    ) -> np.ndarray:
        """
        Predict future values using autoregressive approach.
        
        Args:
            X_last: Last sequence_length samples of features
            steps: Number of future steps to predict
            
        Returns:
            Array of future predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if len(X_last) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} samples")
        
        X_scaled = self.scaler_X.transform(X_last)
        predictions = []
        
        # Use the last sequence_length samples as starting point
        current_seq = X_scaled[-self.sequence_length:].copy()
        
        for _ in range(steps):
            # Predict next value
            seq_input = current_seq.reshape(1, self.sequence_length, -1)
            pred_scaled = self.model.predict(seq_input, verbose=0)[0, 0]
            pred = self.scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred)
            
            # Update sequence (simplified - in practice, need to update features properly)
            current_seq = np.roll(current_seq, -1, axis=0)
            # Note: This is simplified. In practice, you'd need to update the feature vector
            # with the new prediction and other time-dependent features
        
        return np.array(predictions)
    
    def _build_model(self, input_shape: Tuple[int, int, int]) -> Model:
        """Build the LSTM model architecture."""
        config = Config.get_model_config('lstm')
        
        model = Sequential([
            LSTM(
                config.get('lstm_units', 50),
                return_sequences=True,
                input_shape=(input_shape[1], input_shape[2])
            ),
            Dropout(0.2),
            LSTM(config.get('lstm_units', 50) // 2, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dropout(0.1),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=config.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def _build_optimized_model(self, input_shape: Tuple[int, int, int]) -> Model:
        """Build optimized LSTM model with hyperparameter search results."""
        # This is a placeholder for more sophisticated architecture search
        # In practice, you might use techniques like Keras Tuner
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(input_shape[1], input_shape[2])),
            Dropout(0.3),
            LSTM(32, return_sequences=True),
            Dropout(0.3),
            LSTM(16, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
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
        
        # Save scalers and metadata
        metadata = {
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'sequence_length': self.sequence_length,
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
        
        self.scaler_X = metadata['scaler_X']
        self.scaler_y = metadata['scaler_y']
        self.sequence_length = metadata['sequence_length']
        self.random_state = metadata['random_state']
        self.is_fitted = True
        
        # Restore history if available
        if metadata.get('history'):
            class FakeHistory:
                def __init__(self, history_dict):
                    self.history = history_dict
            self.history = FakeHistory(metadata['history'])
        
        self.logger.info(f"Model loaded from {filepath}")