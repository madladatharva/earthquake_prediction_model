"""
Enhanced Random Forest model for earthquake prediction.
Extends the existing implementation with hyperparameter tuning and advanced features.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from typing import Dict, Tuple, Optional, Any
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config

class EnhancedRandomForestModel:
    """Enhanced Random Forest model with hyperparameter optimization."""
    
    def __init__(self, random_state: int = None):
        self.random_state = random_state or Config.RANDOM_STATE
        self.model = None
        self.best_params = None
        self.feature_importance = None
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
        Train the Random Forest model with optional hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training targets  
            optimize_params: Whether to perform hyperparameter optimization
            cv_folds: Number of cross-validation folds
            
        Returns:
            Training results dictionary
        """
        cv_folds = cv_folds or Config.CV_FOLDS
        
        if optimize_params:
            self.model = self._optimize_hyperparameters(X_train, y_train, cv_folds)
        else:
            # Use default parameters from config
            params = Config.get_model_config('random_forest')
            self.model = RandomForestRegressor(**params)
            self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        self.is_fitted = True
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=cv_folds, scoring='r2'
        )
        
        results = {
            'model_type': 'RandomForest',
            'train_metrics': train_metrics,
            'cv_mean_r2': cv_scores.mean(),
            'cv_std_r2': cv_scores.std(),
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }
        
        self.logger.info(f"Random Forest trained: R² = {train_metrics['r2']:.4f}")
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates using tree-level predictions.
        
        Returns:
            Tuple of (predictions, uncertainty_estimates)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        
        # Mean prediction
        predictions = tree_predictions.mean(axis=0)
        
        # Standard deviation as uncertainty estimate  
        uncertainties = tree_predictions.std(axis=0)
        
        return predictions, uncertainties
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """Get feature importance rankings."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def _optimize_hyperparameters(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        cv_folds: int
    ) -> RandomForestRegressor:
        """Optimize hyperparameters using randomized search."""
        
        # Parameter grid for optimization
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Base model
        base_model = RandomForestRegressor(random_state=self.random_state)
        
        # Randomized search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=50,  # Number of random combinations to try
            cv=cv_folds,
            scoring='r2',
            random_state=self.random_state,
            n_jobs=Config.get_safe_n_jobs(),  # Use safe n_jobs for Windows compatibility
            verbose=0
        )
        
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
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a saved model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.best_params = model_data['best_params']
        self.feature_importance = model_data['feature_importance'] 
        self.random_state = model_data['random_state']
        self.is_fitted = True
        
        self.logger.info(f"Model loaded from {filepath}")