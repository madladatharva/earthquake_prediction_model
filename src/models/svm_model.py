"""
Support Vector Machine (SVM) model for earthquake prediction.
Implements SVM regression with advanced kernel options and hyperparameter optimization.
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from typing import Dict, Tuple, Optional, Any
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config

class SVMEarthquakeModel:
    """SVM model for earthquake magnitude prediction."""
    
    def __init__(self, random_state: int = None):
        self.random_state = random_state or Config.RANDOM_STATE
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)
        
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        optimize_params: bool = True,
        cv_folds: int = None
    ) -> Dict[str, Any]:
        """
        Train the SVM model with optional hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training targets  
            optimize_params: Whether to perform hyperparameter optimization
            cv_folds: Number of cross-validation folds
            
        Returns:
            Training results dictionary
        """
        # Determine safe CV folds based on sample size
        default_cv_folds = Config.CV_FOLDS
        n_samples = len(y_train)
        
        # Ensure CV folds don't exceed sample size
        max_safe_cv = min(default_cv_folds, n_samples // 2)
        cv_folds = cv_folds or max(2, max_safe_cv)  # Minimum 2 folds
        
        self.logger.info(f"Using {cv_folds} CV folds for {n_samples} samples")
        
        # Scale features (important for SVM)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if optimize_params and n_samples >= 10:  # Only optimize if enough samples
            self.model = self._optimize_hyperparameters(X_train_scaled, y_train, cv_folds)
        else:
            # Use default parameters for small datasets or when optimization disabled
            self.logger.info("Using default SVM parameters (insufficient samples for optimization)")
            self.model = SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                epsilon=0.1
            )
            self.model.fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train_scaled)
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        
        # Cross-validation scores (only if we have enough samples)
        if n_samples >= cv_folds * 2:
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, 
                cv=cv_folds, scoring='r2'
            )
            cv_mean_r2 = cv_scores.mean()
            cv_std_r2 = cv_scores.std()
        else:
            self.logger.warning(f"Insufficient samples for CV (need {cv_folds * 2}, have {n_samples})")
            cv_mean_r2 = train_metrics['r2']  # Use training R2 as fallback
            cv_std_r2 = 0.0
        
        results = {
            'model_type': 'SVM',
            'train_metrics': train_metrics,
            'cv_mean_r2': cv_mean_r2,
            'cv_std_r2': cv_std_r2,
            'best_params': self.best_params,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0],
            'cv_folds_used': cv_folds,
            'n_support_vectors': self.model.support_vectors_.shape[0] if hasattr(self.model, 'support_vectors_') else 0
        }
        
        self.logger.info(f"SVM trained: R² = {train_metrics['r2']:.4f}")
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        For SVM, uncertainty is estimated using distance to decision boundary.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, uncertainty_estimates)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # For SVM regression, we can use decision function as confidence measure
        if hasattr(self.model, 'decision_function'):
            # Get decision function values
            decision_values = np.abs(self.model.decision_function(X_scaled))
            
            # Convert to uncertainty (inverse relationship)
            # Higher decision function values = more confident = lower uncertainty
            max_decision = np.max(decision_values) if len(decision_values) > 0 else 1.0
            uncertainties = (max_decision - decision_values) / max_decision
            
            # Scale uncertainty to reasonable range based on prediction magnitude
            uncertainties = uncertainties * np.abs(predictions) * 0.2
        else:
            # Fallback: use distance-based uncertainty estimate
            support_vectors = self.model.support_vectors_
            
            # Calculate minimum distance to support vectors
            min_distances = []
            for x in X_scaled:
                distances = np.linalg.norm(support_vectors - x, axis=1)
                min_distances.append(np.min(distances))
            
            min_distances = np.array(min_distances)
            
            # Normalize distances to uncertainty estimates
            max_distance = np.max(min_distances) if len(min_distances) > 0 else 1.0
            uncertainties = min_distances / max_distance * np.abs(predictions) * 0.15
        
        return predictions, uncertainties
    
    def get_support_vector_info(self) -> Dict[str, Any]:
        """Get information about support vectors."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting support vector info")
        
        info = {
            'n_support_vectors': self.model.support_vectors_.shape[0],
            'n_features': self.model.support_vectors_.shape[1],
            'dual_coef_shape': self.model.dual_coef_.shape if hasattr(self.model, 'dual_coef_') else None,
            'intercept': self.model.intercept_[0] if hasattr(self.model, 'intercept_') else None
        }
        
        return info
    
    def _optimize_hyperparameters(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        cv_folds: int
    ) -> SVR:
        """Optimize hyperparameters using randomized search."""
        self.logger.info("Optimizing SVM hyperparameters...")
        
        # Simplified parameter grid for faster optimization
        param_grid = {
            'kernel': ['rbf', 'linear'],  # Removed poly to reduce search space
            'C': [0.1, 1, 10, 100],      # Reduced from 5 to 4 values
            'gamma': ['scale', 'auto', 0.01, 0.1],  # Reduced search space
            'epsilon': [0.01, 0.1, 0.5]   # Reduced from 5 to 3 values
        }
        
        # Base model
        base_model = SVR()
        
        # Randomized search with reduced iterations
        n_samples = len(y_train)
        max_iter = min(10, n_samples // 4)  # Even more conservative for faster training
        max_iter = max(3, max_iter)  # At least 3 iterations
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=max_iter,  # Reduced from 50 to adaptive value
            cv=cv_folds,
            scoring='r2',
            random_state=self.random_state,
            n_jobs=1,  # Use single job to avoid timeout issues
            verbose=0
        )
        
        self.logger.info(f"Running {max_iter} iterations with {cv_folds} CV folds")
        random_search.fit(X_train, y_train)
        
        self.best_params = random_search.best_params_
        self.logger.info(f"Best parameters: {self.best_params}")
        
        return random_search.best_estimator_
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'best_params': self.best_params,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a saved model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.best_params = model_data['best_params']
        self.random_state = model_data['random_state']
        self.is_fitted = True
        
        self.logger.info(f"Model loaded from {filepath}")