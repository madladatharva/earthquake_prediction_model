"""
Ensemble model that combines multiple ML models for earthquake prediction.
Uses weighted averaging and stacking approaches for improved predictions.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from typing import Dict, List, Tuple, Optional, Any
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config

# Import individual models
from .random_forest_model import EnhancedRandomForestModel
from .xgboost_model import XGBoostModel
from .neural_network_model import EnhancedNeuralNetworkModel
from .svm_model import SVMEarthquakeModel

class EarthquakeEnsembleModel:
    """Ensemble model combining multiple ML approaches for earthquake prediction."""
    
    def __init__(self, random_state: int = None):
        self.random_state = random_state or Config.RANDOM_STATE
        self.models = {}
        self.ensemble_weights = Config.ENSEMBLE_WEIGHTS.copy()
        self.meta_model = None
        self.is_fitted = False
        self.ensemble_type = 'weighted_average'  # 'weighted_average' or 'stacking'
        self.logger = logging.getLogger(__name__)
        
    def initialize_models(self, include_models: Optional[List[str]] = None) -> None:
        """
        Initialize individual models in the ensemble.
        
        Args:
            include_models: List of models to include. If None, includes all available models.
        """
        available_models = ['random_forest', 'xgboost', 'neural_network', 'svm']
        
        if include_models is None:
            include_models = available_models
        
        # Initialize requested models
        if 'random_forest' in include_models:
            self.models['random_forest'] = EnhancedRandomForestModel(self.random_state)
            
        if 'xgboost' in include_models:
            self.models['xgboost'] = XGBoostModel(self.random_state)
            
        if 'neural_network' in include_models:
            self.models['neural_network'] = EnhancedNeuralNetworkModel(self.random_state)
            
        if 'svm' in include_models:
            self.models['svm'] = SVMEarthquakeModel(self.random_state)
        
        # Update weights to only include active models
        active_weights = {k: v for k, v in self.ensemble_weights.items() if k in self.models}
        total_weight = sum(active_weights.values())
        
        # Normalize weights
        if total_weight > 0:
            self.ensemble_weights = {k: v/total_weight for k, v in active_weights.items()}
        
        self.logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        ensemble_type: str = 'weighted_average',
        optimize_weights: bool = True,
        cv_folds: int = None
    ) -> Dict[str, Any]:
        """
        Train the ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            ensemble_type: 'weighted_average' or 'stacking'
            optimize_weights: Whether to optimize ensemble weights
            cv_folds: Number of cross-validation folds
            
        Returns:
            Training results dictionary
        """
        if not self.models:
            raise ValueError("No models initialized. Call initialize_models() first.")
        
        cv_folds = cv_folds or Config.CV_FOLDS
        self.ensemble_type = ensemble_type
        
        # Train individual models
        model_results = {}
        individual_predictions = {}
        models_to_remove = []
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name} model...")
            
            try:
                if name == 'neural_network':
                    # Neural network can use validation data
                    result = model.train(X_train, y_train, X_val, y_val)
                elif name == 'lstm':
                    # LSTM can use validation data  
                    result = model.train(X_train, y_train, X_val, y_val)
                else:
                    # Other models use standard training
                    result = model.train(X_train, y_train)
                
                model_results[name] = result
                
                # Get predictions for ensemble training
                individual_predictions[name] = model.predict(X_train)
                
                self.logger.info(f"{name} trained successfully: R² = {result['train_metrics']['r2']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train {name}: {e}")
                # Mark for removal
                models_to_remove.append(name)
        
        # Remove failed models
        for name in models_to_remove:
            if name in self.models:
                del self.models[name]
            if name in self.ensemble_weights:
                del self.ensemble_weights[name]
        
        if not self.models:
            raise ValueError("All models failed to train")
        
        # Optimize ensemble weights or train meta-model
        if ensemble_type == 'weighted_average':
            if optimize_weights:
                self.ensemble_weights = self._optimize_weights(individual_predictions, y_train)
        elif ensemble_type == 'stacking':
            self._train_meta_model(individual_predictions, y_train, cv_folds)
        
        self.is_fitted = True
        
        # Calculate ensemble metrics
        ensemble_pred = self.predict(X_train)
        ensemble_metrics = self._calculate_metrics(y_train, ensemble_pred)
        
        # Cross-validation for ensemble
        cv_scores = self._ensemble_cross_validation(X_train, y_train, cv_folds)
        
        results = {
            'model_type': f'Ensemble_{ensemble_type}',
            'individual_models': list(self.models.keys()),
            'ensemble_metrics': ensemble_metrics,
            'cv_mean_r2': cv_scores.mean(),
            'cv_std_r2': cv_scores.std(),
            'ensemble_weights': self.ensemble_weights,
            'individual_results': model_results,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred)
            results['val_metrics'] = val_metrics
        
        self.logger.info(f"Ensemble trained: R² = {ensemble_metrics['r2']:.4f}")
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be trained before making predictions")
        
        if self.ensemble_type == 'weighted_average':
            return self._predict_weighted_average(X)
        elif self.ensemble_type == 'stacking':
            return self._predict_stacking(X)
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        Combines individual model uncertainties and ensemble disagreement.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, uncertainty_estimates)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be trained before making predictions")
        
        individual_predictions = []
        individual_uncertainties = []
        
        # Get predictions and uncertainties from each model
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_with_uncertainty'):
                    pred, unc = model.predict_with_uncertainty(X)
                else:
                    pred = model.predict(X)
                    unc = np.abs(pred) * 0.1  # Fallback uncertainty
                
                individual_predictions.append(pred)
                individual_uncertainties.append(unc)
                
            except Exception as e:
                self.logger.warning(f"Failed to get uncertainty from {name}: {e}")
        
        individual_predictions = np.array(individual_predictions)
        individual_uncertainties = np.array(individual_uncertainties)
        
        # Ensemble prediction
        if self.ensemble_type == 'weighted_average':
            weights = np.array([self.ensemble_weights.get(name, 0) for name in self.models.keys()])
            weights = weights[:, np.newaxis]  # Shape for broadcasting
            
            ensemble_pred = np.sum(weights * individual_predictions, axis=0)
            
            # Ensemble uncertainty: combination of individual uncertainties and disagreement
            weighted_uncertainties = np.sum(weights * individual_uncertainties, axis=0)
            disagreement = np.std(individual_predictions, axis=0)
            ensemble_unc = np.sqrt(weighted_uncertainties**2 + disagreement**2)
            
        else:  # stacking
            ensemble_pred = self._predict_stacking(X)
            
            # For stacking, use disagreement among base models as uncertainty
            disagreement = np.std(individual_predictions, axis=0)
            ensemble_unc = disagreement
        
        return ensemble_pred, ensemble_unc
    
    def _predict_weighted_average(self, X: np.ndarray) -> np.ndarray:
        """Make weighted average predictions."""
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            weight = self.ensemble_weights.get(name, 0)
            
            predictions.append(pred * weight)
            weights.append(weight)
        
        return np.sum(predictions, axis=0)
    
    def _predict_stacking(self, X: np.ndarray) -> np.ndarray:
        """Make stacking predictions using meta-model."""
        if self.meta_model is None:
            raise ValueError("Meta-model not trained for stacking")
        
        # Get base model predictions
        base_predictions = []
        for model in self.models.values():
            pred = model.predict(X)
            base_predictions.append(pred)
        
        # Stack predictions as features for meta-model
        base_predictions = np.column_stack(base_predictions)
        
        return self.meta_model.predict(base_predictions)
    
    def _optimize_weights(
        self, 
        individual_predictions: Dict[str, np.ndarray], 
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """Optimize ensemble weights to minimize error."""
        from scipy.optimize import minimize
        
        model_names = list(individual_predictions.keys())
        pred_matrix = np.column_stack([individual_predictions[name] for name in model_names])
        
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize weights
            ensemble_pred = np.sum(pred_matrix * weights, axis=1)
            return mean_squared_error(y_true, ensemble_pred)
        
        # Initial weights (equal)
        n_models = len(model_names)
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(
            objective, 
            initial_weights, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Return optimized weights as dictionary
        optimized_weights = {name: weight for name, weight in zip(model_names, result.x)}
        
        self.logger.info(f"Optimized weights: {optimized_weights}")
        return optimized_weights
    
    def _train_meta_model(
        self, 
        individual_predictions: Dict[str, np.ndarray], 
        y_true: np.ndarray,
        cv_folds: int
    ) -> None:
        """Train meta-model for stacking ensemble."""
        # Prepare features for meta-model (base model predictions)
        base_predictions = np.column_stack([pred for pred in individual_predictions.values()])
        
        # Use linear regression as meta-model (simple and interpretable)
        self.meta_model = LinearRegression()
        self.meta_model.fit(base_predictions, y_true)
        
        # Cross-validation score for meta-model
        cv_score = cross_val_score(
            self.meta_model, base_predictions, y_true,
            cv=cv_folds, scoring='r2'
        ).mean()
        
        self.logger.info(f"Meta-model trained: CV R² = {cv_score:.4f}")
    
    def _ensemble_cross_validation(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        cv_folds: int
    ) -> np.ndarray:
        """Perform cross-validation for the ensemble."""
        from sklearn.model_selection import KFold
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []
        
        for train_idx, val_idx in kfold.split(X):
            # Fix pandas indexing issue: use iloc for row selection with integer indices
            if hasattr(X, 'iloc'):
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Create temporary ensemble for this fold
            temp_ensemble = EarthquakeEnsembleModel(self.random_state)
            temp_ensemble.initialize_models(list(self.models.keys()))
            
            try:
                temp_ensemble.train(
                    X_fold_train, y_fold_train,
                    ensemble_type=self.ensemble_type,
                    optimize_weights=False  # Skip optimization for speed
                )
                
                y_pred = temp_ensemble.predict(X_fold_val)
                r2 = r2_score(y_fold_val, y_pred)
                cv_scores.append(r2)
                
            except Exception as e:
                self.logger.warning(f"CV fold failed: {e}")
        
        return np.array(cv_scores)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def get_model_contributions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get individual model contributions to ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be trained before getting contributions")
        
        contributions = {}
        for name, model in self.models.items():
            pred = model.predict(X)
            weight = self.ensemble_weights.get(name, 0)
            contributions[name] = pred * weight
        
        return contributions
    
    def save_ensemble(self, filepath: str) -> None:
        """Save the entire ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be trained before saving")
        
        # Save individual models
        model_paths = {}
        for name, model in self.models.items():
            model_path = filepath.replace('.pkl', f'_{name}_model.pkl')
            model.save_model(model_path)
            model_paths[name] = model_path
        
        # Save ensemble metadata
        ensemble_data = {
            'ensemble_weights': self.ensemble_weights,
            'ensemble_type': self.ensemble_type,
            'meta_model': self.meta_model,
            'model_paths': model_paths,
            'random_state': self.random_state
        }
        
        joblib.dump(ensemble_data, filepath)
        self.logger.info(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str) -> None:
        """Load a saved ensemble."""
        ensemble_data = joblib.load(filepath)
        
        # Load individual models
        self.models = {}
        for name, model_path in ensemble_data['model_paths'].items():
            if name == 'random_forest':
                model = EnhancedRandomForestModel(self.random_state)
            elif name == 'xgboost':
                model = XGBoostModel(self.random_state)
            elif name == 'neural_network':
                model = EnhancedNeuralNetworkModel(self.random_state)
            elif name == 'svm':
                model = SVMEarthquakeModel(self.random_state)
            else:
                continue
            
            model.load_model(model_path)
            self.models[name] = model
        
        # Load ensemble metadata
        self.ensemble_weights = ensemble_data['ensemble_weights']
        self.ensemble_type = ensemble_data['ensemble_type']
        self.meta_model = ensemble_data['meta_model']
        self.random_state = ensemble_data['random_state']
        self.is_fitted = True
        
        self.logger.info(f"Ensemble loaded from {filepath}")