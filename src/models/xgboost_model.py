"""
XGBoost model for earthquake prediction.
Implements gradient boosting with advanced hyperparameter optimization.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from typing import Dict, Tuple, Optional, Any
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config

class XGBoostModel:
    """XGBoost model for earthquake magnitude prediction."""
    
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
        cv_folds: int = None,
        early_stopping_rounds: int = 10
    ) -> Dict[str, Any]:
        """
        Train the XGBoost model with optional hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training targets  
            optimize_params: Whether to perform hyperparameter optimization
            cv_folds: Number of cross-validation folds
            early_stopping_rounds: Early stopping rounds for training
            
        Returns:
            Training results dictionary
        """
        cv_folds = cv_folds or Config.CV_FOLDS
        
        if optimize_params:
            self.model = self._optimize_hyperparameters(X_train, y_train, cv_folds)
        else:
            # Use default parameters from config
            params = Config.get_model_config('xgboost')
            self.model = xgb.XGBRegressor(**params)
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
            'model_type': 'XGBoost',
            'train_metrics': train_metrics,
            'cv_mean_r2': cv_scores.mean(),
            'cv_std_r2': cv_scores.std(),
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }
        
        self.logger.info(f"XGBoost trained: R² = {train_metrics['r2']:.4f}")
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X: np.ndarray, n_estimators_sample: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates using subsampled trees.
        
        Args:
            X: Input features
            n_estimators_sample: Number of estimators to sample for uncertainty
            
        Returns:
            Tuple of (predictions, uncertainty_estimates)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # For XGBoost, we'll use model intrinsic uncertainty through tree sampling
        predictions = self.model.predict(X)
        
        # Approximate uncertainty using prediction variance
        # This is a simplified approach - in production, consider using quantile regression
        if hasattr(self.model, 'get_booster'):
            # Use XGBoost's built-in prediction with iteration_range for sampling
            try:
                n_trees = min(n_estimators_sample, self.model.n_estimators)
                pred_samples = []
                
                for i in range(5):  # Sample 5 different tree subsets
                    iteration_end = max(1, n_trees - np.random.randint(0, n_trees // 3))
                    pred_i = self.model.get_booster().predict(
                        xgb.DMatrix(X), 
                        iteration_range=(0, iteration_end)
                    )
                    pred_samples.append(pred_i)
                
                pred_samples = np.array(pred_samples)
                uncertainties = pred_samples.std(axis=0)
            except Exception:
                # Fallback to simple heuristic
                uncertainties = np.abs(predictions) * 0.1
        else:
            # Fallback: use a simple heuristic based on prediction confidence
            uncertainties = np.abs(predictions) * 0.1  # 10% uncertainty heuristic
        
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
    
    def get_shap_values(self, X: np.ndarray, feature_names: Optional[list] = None) -> Dict[str, Any]:
        """
        Get SHAP values for model interpretability (if shap is available).
        
        Args:
            X: Input features for SHAP calculation
            feature_names: Names of features
            
        Returns:
            Dictionary with SHAP values and related info
        """
        try:
            import shap
            
            if not self.is_fitted:
                raise ValueError("Model must be trained before getting SHAP values")
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            
            return {
                'shap_values': shap_values,
                'expected_value': explainer.expected_value,
                'feature_names': feature_names or [f'feature_{i}' for i in range(X.shape[1])]
            }
        except ImportError:
            self.logger.warning("SHAP not available. Install with: pip install shap")
            return {}
    
    def _optimize_hyperparameters(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        cv_folds: int
    ) -> xgb.XGBRegressor:
        """Optimize hyperparameters using randomized search."""
        
        # Parameter grid for optimization
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }
        
        # Base model
        base_model = xgb.XGBRegressor(
            random_state=self.random_state,
            tree_method='hist',  # Faster training
            eval_metric='rmse'
        )
        
        # Randomized search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=30,  # Number of random combinations to try
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
        
        # XGBoost models can be saved in their native format
        model_path = filepath.replace('.pkl', '.xgb')
        self.model.save_model(model_path)
        
        # Save additional metadata
        metadata = {
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'random_state': self.random_state,
            'model_path': model_path
        }
        
        joblib.dump(metadata, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a saved model."""
        metadata = joblib.load(filepath)
        
        # Load the XGBoost model
        model_path = metadata['model_path']
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        
        self.best_params = metadata['best_params']
        self.feature_importance = metadata['feature_importance'] 
        self.random_state = metadata['random_state']
        self.is_fitted = True
        
        self.logger.info(f"Model loaded from {filepath}")