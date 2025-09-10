"""
Comprehensive model evaluation framework for earthquake prediction models.
Includes cross-validation, backtesting, and performance monitoring capabilities.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    mean_absolute_percentage_error
)
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config

class EarthquakeModelEvaluator:
    """Comprehensive model evaluation and comparison framework."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.evaluation_results = {}
        
    def evaluate_model(
        self,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model: Trained model object
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            model_name: Name of the model for identification
            feature_names: Names of features (optional)
            
        Returns:
            Dictionary containing evaluation results
        """
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Basic predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Basic metrics
            results['train_metrics'] = self._calculate_regression_metrics(y_train, y_train_pred)
            results['test_metrics'] = self._calculate_regression_metrics(y_test, y_test_pred)
            
            # Cross-validation
            results['cv_results'] = self._cross_validation_evaluation(model, X_train, y_train)
            
            # Prediction intervals (if model supports uncertainty)
            if hasattr(model, 'predict_with_uncertainty'):
                results['uncertainty_metrics'] = self._evaluate_uncertainty(
                    model, X_test, y_test
                )
            
            # Residual analysis
            results['residual_analysis'] = self._analyze_residuals(y_test, y_test_pred)
            
            # Feature importance (if available)
            if hasattr(model, 'get_feature_importance') and feature_names:
                results['feature_importance'] = self._analyze_feature_importance(
                    model, feature_names
                )
            
            # Model complexity metrics
            results['complexity_metrics'] = self._calculate_complexity_metrics(model)
            
            # Performance summary
            results['performance_summary'] = self._create_performance_summary(results)
            
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_name}: {e}")
            results['error'] = str(e)
        
        # Store results
        self.evaluation_results[model_name] = results
        
        return results
    
    def compare_models(
        self,
        model_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare multiple models and create comparison report.
        
        Args:
            model_results: Dictionary of model evaluation results
            
        Returns:
            Model comparison results
        """
        if not model_results:
            return {}
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'n_models': len(model_results),
            'models': list(model_results.keys())
        }
        
        # Extract key metrics for comparison
        metrics_comparison = {}
        
        for metric_name in ['r2', 'rmse', 'mae', 'mape']:
            test_scores = []
            cv_scores = []
            
            for model_name, results in model_results.items():
                if 'error' not in results:
                    # Test scores
                    if metric_name in results.get('test_metrics', {}):
                        test_scores.append({
                            'model': model_name,
                            'score': results['test_metrics'][metric_name]
                        })
                    
                    # CV scores
                    if metric_name in results.get('cv_results', {}):
                        cv_scores.append({
                            'model': model_name,
                            'score': results['cv_results'][metric_name]['mean']
                        })
            
            if test_scores:
                # Sort by score (higher is better for R², lower is better for error metrics)
                reverse_sort = metric_name == 'r2'
                test_scores.sort(key=lambda x: x['score'], reverse=reverse_sort)
                
                metrics_comparison[f'{metric_name}_test'] = {
                    'rankings': test_scores,
                    'best_model': test_scores[0]['model'],
                    'best_score': test_scores[0]['score']
                }
            
            if cv_scores:
                cv_scores.sort(key=lambda x: x['score'], reverse=reverse_sort)
                
                metrics_comparison[f'{metric_name}_cv'] = {
                    'rankings': cv_scores,
                    'best_model': cv_scores[0]['model'],
                    'best_score': cv_scores[0]['score']
                }
        
        comparison['metrics_comparison'] = metrics_comparison
        
        # Overall ranking (based on test R²)
        if 'r2_test' in metrics_comparison:
            comparison['overall_ranking'] = metrics_comparison['r2_test']['rankings']
        
        # Statistical significance tests (if multiple models)
        if len(model_results) >= 2:
            comparison['statistical_tests'] = self._perform_statistical_tests(model_results)
        
        return comparison
    
    def backtest_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        time_column: Optional[np.ndarray] = None,
        n_splits: int = 5,
        test_size_ratio: float = 0.2
    ) -> Dict[str, Any]:
        """
        Perform backtesting with temporal splits.
        
        Args:
            model: Model to backtest
            X: Features
            y: Targets
            time_column: Time values for temporal ordering
            n_splits: Number of temporal splits
            test_size_ratio: Ratio of data to use for testing in each split
            
        Returns:
            Backtesting results
        """
        if time_column is not None:
            # Sort by time
            sort_indices = np.argsort(time_column)
            # Fix pandas indexing issue: use iloc for row selection with integer indices
            if hasattr(X, 'iloc'):
                X_sorted = X.iloc[sort_indices]
                y_sorted = y.iloc[sort_indices] if hasattr(y, 'iloc') else y[sort_indices]
            else:
                X_sorted = X[sort_indices]
                y_sorted = y[sort_indices]
        else:
            # Use data as-is
            X_sorted = X
            y_sorted = y
        
        backtest_results = {
            'n_splits': n_splits,
            'test_size_ratio': test_size_ratio,
            'split_results': []
        }
        
        # Create temporal splits
        total_samples = len(X_sorted)
        test_size = int(total_samples * test_size_ratio)
        
        for i in range(n_splits):
            # Calculate split indices
            end_idx = total_samples - (n_splits - 1 - i) * (test_size // n_splits)
            start_idx = end_idx - test_size
            train_end_idx = start_idx
            
            if train_end_idx <= test_size:  # Not enough training data
                continue
            
            # Split data
            # Fix pandas indexing issue: use iloc for row selection with integer indices
            if hasattr(X_sorted, 'iloc'):
                X_train_bt = X_sorted.iloc[:train_end_idx]
                X_test_bt = X_sorted.iloc[start_idx:end_idx]
                y_train_bt = y_sorted.iloc[:train_end_idx] if hasattr(y_sorted, 'iloc') else y_sorted[:train_end_idx]
                y_test_bt = y_sorted.iloc[start_idx:end_idx] if hasattr(y_sorted, 'iloc') else y_sorted[start_idx:end_idx]
            else:
                X_train_bt = X_sorted[:train_end_idx]
                X_test_bt = X_sorted[start_idx:end_idx]
                y_train_bt = y_sorted[:train_end_idx]
                y_test_bt = y_sorted[start_idx:end_idx]
            
            try:
                # Train model on historical data
                if hasattr(model, 'train'):
                    # Custom model with train method
                    model.train(X_train_bt, y_train_bt)
                else:
                    # Sklearn-style model
                    model.fit(X_train_bt, y_train_bt)
                
                # Predict on future data
                y_pred_bt = model.predict(X_test_bt)
                
                # Calculate metrics
                split_metrics = self._calculate_regression_metrics(y_test_bt, y_pred_bt)
                
                split_result = {
                    'split_id': i,
                    'train_size': len(X_train_bt),
                    'test_size': len(X_test_bt),
                    'metrics': split_metrics
                }
                
                backtest_results['split_results'].append(split_result)
                
            except Exception as e:
                self.logger.warning(f"Backtest split {i} failed: {e}")
        
        # Aggregate results
        if backtest_results['split_results']:
            backtest_results['aggregated_metrics'] = self._aggregate_backtest_results(
                backtest_results['split_results']
            )
        
        return backtest_results
    
    def _calculate_regression_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'max_error': np.max(np.abs(y_true - y_pred)),
            'mean_error': np.mean(y_true - y_pred),  # Bias
            'std_error': np.std(y_true - y_pred)     # Variance of residuals
        }
    
    def _cross_validation_evaluation(
        self, 
        model, 
        X: np.ndarray, 
        y: np.ndarray,
        cv_folds: int = None
    ) -> Dict[str, Any]:
        """Perform cross-validation evaluation with adaptive fold handling."""
        cv_folds = cv_folds or Config.CV_FOLDS
        
        # Adaptive cross-validation: adjust folds based on sample size
        n_samples = len(X)
        
        # Ensure we have enough samples for cross-validation
        # TimeSeriesSplit requires at least (n_splits + 1) samples
        max_folds = min(cv_folds, max(2, n_samples // 3))  # At least 3 samples per fold
        
        if max_folds < 2:
            self.logger.warning(f"Insufficient data for cross-validation: {n_samples} samples. Skipping CV.")
            return {
                'error': f'Insufficient data for cross-validation: {n_samples} samples (minimum 6 required)',
                'n_samples': n_samples,
                'required_min_samples': 6
            }
        
        # Use the adaptive number of folds
        actual_folds = max_folds
        self.logger.info(f"Using {actual_folds} CV folds for {n_samples} samples (requested: {cv_folds})")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=actual_folds)
        
        cv_results = {
            'n_splits_requested': cv_folds,
            'n_splits_used': actual_folds,
            'n_samples': n_samples
        }
        
        # Different metrics
        scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(model, X, y, cv=tscv, scoring=metric)
                
                # Convert negative metrics back to positive
                if 'neg_' in metric:
                    scores = -scores
                    metric = metric.replace('neg_', '')
                
                cv_results[metric] = {
                    'scores': scores.tolist(),
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'min': scores.min(),
                    'max': scores.max()
                }
                
            except Exception as e:
                self.logger.warning(f"CV metric {metric} failed: {e}")
                cv_results[f'{metric}_error'] = str(e)
        
        return cv_results
    
    def _evaluate_uncertainty(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate uncertainty estimation quality."""
        try:
            predictions, uncertainties = model.predict_with_uncertainty(X_test)
            
            # Calibration: check if uncertainties are well-calibrated
            residuals = np.abs(y_test - predictions)
            
            uncertainty_results = {
                'mean_uncertainty': np.mean(uncertainties),
                'std_uncertainty': np.std(uncertainties),
                'uncertainty_range': [np.min(uncertainties), np.max(uncertainties)],
                'correlation_with_error': np.corrcoef(uncertainties, residuals)[0, 1]
            }
            
            # Coverage probabilities for different confidence levels
            confidence_levels = [0.68, 0.90, 0.95, 0.99]
            
            for conf_level in confidence_levels:
                z_score = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[conf_level]
                
                # Count how many true values fall within confidence interval
                lower_bound = predictions - z_score * uncertainties
                upper_bound = predictions + z_score * uncertainties
                
                coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
                uncertainty_results[f'coverage_{int(conf_level*100)}%'] = coverage
            
            return uncertainty_results
            
        except Exception as e:
            self.logger.warning(f"Uncertainty evaluation failed: {e}")
            return {}
    
    def _analyze_residuals(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze residuals for model diagnostics."""
        residuals = y_true - y_pred
        
        return {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'skewness': float(pd.Series(residuals).skew()),
            'kurtosis': float(pd.Series(residuals).kurtosis()),
            'normality_test_p_value': self._test_normality(residuals),
            'outlier_count': np.sum(np.abs(residuals) > 2 * np.std(residuals)),
            'outlier_percentage': np.mean(np.abs(residuals) > 2 * np.std(residuals)) * 100
        }
    
    def _analyze_feature_importance(
        self, 
        model, 
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze feature importance from the model."""
        try:
            importance_df = model.get_feature_importance(feature_names)
            
            return {
                'top_10_features': importance_df.head(10).to_dict('records'),
                'feature_importance_stats': {
                    'max_importance': importance_df['importance'].max(),
                    'mean_importance': importance_df['importance'].mean(),
                    'std_importance': importance_df['importance'].std(),
                    'n_zero_importance': (importance_df['importance'] == 0).sum()
                }
            }
        except Exception as e:
            self.logger.warning(f"Feature importance analysis failed: {e}")
            return {}
    
    def _calculate_complexity_metrics(self, model) -> Dict[str, Any]:
        """Calculate model complexity metrics."""
        complexity = {}
        
        # Try to get model-specific complexity measures
        try:
            if hasattr(model, 'n_estimators'):  # Tree-based models
                complexity['n_estimators'] = model.n_estimators
            
            if hasattr(model, 'max_depth'):
                complexity['max_depth'] = model.max_depth
            
            if hasattr(model, 'coef_'):  # Linear models
                complexity['n_parameters'] = len(model.coef_)
            
            if hasattr(model, 'support_vectors_'):  # SVM
                complexity['n_support_vectors'] = len(model.support_vectors_)
            
            if hasattr(model, 'model') and hasattr(model.model, 'count_params'):  # Neural networks
                complexity['n_parameters'] = model.model.count_params()
                
        except Exception as e:
            self.logger.warning(f"Complexity calculation failed: {e}")
        
        return complexity
    
    def _create_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of model performance."""
        summary = {}
        
        # Test performance
        if 'test_metrics' in results:
            test_r2 = results['test_metrics'].get('r2', 0)
            test_rmse = results['test_metrics'].get('rmse', float('inf'))
            
            # Performance categories based on R²
            if test_r2 >= 0.9:
                performance_category = 'excellent'
            elif test_r2 >= 0.8:
                performance_category = 'good'
            elif test_r2 >= 0.6:
                performance_category = 'fair'
            else:
                performance_category = 'poor'
            
            summary.update({
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'performance_category': performance_category
            })
        
        # CV stability
        if 'cv_results' in results and 'r2' in results['cv_results']:
            cv_std = results['cv_results']['r2']['std']
            cv_stability = 'stable' if cv_std < 0.1 else 'unstable'
            
            summary.update({
                'cv_stability': cv_stability,
                'cv_r2_std': cv_std
            })
        
        # Uncertainty quality
        if 'uncertainty_metrics' in results:
            uncertainty_corr = results['uncertainty_metrics'].get('correlation_with_error', 0)
            uncertainty_quality = 'good' if uncertainty_corr > 0.5 else 'poor'
            
            summary['uncertainty_quality'] = uncertainty_quality
        
        return summary
    
    def _test_normality(self, residuals: np.ndarray) -> float:
        """Test normality of residuals using Shapiro-Wilk test."""
        try:
            from scipy.stats import shapiro
            _, p_value = shapiro(residuals[:min(5000, len(residuals))])  # Limit sample size
            return float(p_value)
        except ImportError:
            return 0.0  # Assume non-normal if scipy not available
    
    def _perform_statistical_tests(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Perform statistical tests to compare models."""
        # This is a simplified implementation
        # In practice, you might use paired t-tests, Wilcoxon tests, etc.
        
        tests = {
            'note': 'Statistical significance testing requires paired predictions from CV'
        }
        
        # For now, just report which model is best by different metrics
        best_models = {}
        
        for model_name, results in model_results.items():
            if 'error' not in results and 'test_metrics' in results:
                r2 = results['test_metrics'].get('r2', -999)
                
                if 'r2' not in best_models or r2 > best_models['r2']['score']:
                    best_models['r2'] = {'model': model_name, 'score': r2}
        
        tests['best_by_metric'] = best_models
        
        return tests
    
    def _aggregate_backtest_results(self, split_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from multiple backtest splits."""
        metrics = ['r2', 'rmse', 'mae', 'mape']
        
        aggregated = {}
        
        for metric in metrics:
            values = [split['metrics'][metric] for split in split_results if metric in split['metrics']]
            
            if values:
                aggregated[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        return aggregated
    
    def generate_evaluation_report(
        self, 
        model_results: Dict[str, Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """Generate a comprehensive evaluation report."""
        
        report = []
        report.append("# Earthquake Prediction Model Evaluation Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of Models Evaluated: {len(model_results)}")
        report.append("")
        
        # Model comparison
        if len(model_results) > 1:
            comparison = self.compare_models(model_results)
            
            report.append("## Model Comparison Summary")
            
            if 'overall_ranking' in comparison:
                report.append("### Overall Performance Ranking (by Test R²)")
                for i, model_info in enumerate(comparison['overall_ranking']):
                    report.append(f"{i+1}. {model_info['model']}: R² = {model_info['score']:.4f}")
                report.append("")
        
        # Individual model details
        report.append("## Individual Model Results")
        
        for model_name, results in model_results.items():
            if 'error' in results:
                report.append(f"### {model_name} - FAILED")
                report.append(f"Error: {results['error']}")
                report.append("")
                continue
            
            report.append(f"### {model_name}")
            
            # Performance summary
            if 'performance_summary' in results:
                summary = results['performance_summary']
                report.append(f"**Performance Category**: {summary.get('performance_category', 'unknown').title()}")
                report.append(f"**Test R²**: {summary.get('test_r2', 'N/A'):.4f}")
                report.append(f"**Test RMSE**: {summary.get('test_rmse', 'N/A'):.4f}")
                
                if 'cv_stability' in summary:
                    report.append(f"**CV Stability**: {summary['cv_stability'].title()}")
            
            # Test metrics
            if 'test_metrics' in results:
                report.append("**Test Metrics:**")
                for metric, value in results['test_metrics'].items():
                    report.append(f"- {metric.upper()}: {value:.4f}")
            
            # Feature importance (top 5)
            if 'feature_importance' in results and 'top_10_features' in results['feature_importance']:
                report.append("**Top 5 Most Important Features:**")
                for i, feat in enumerate(results['feature_importance']['top_10_features'][:5]):
                    report.append(f"{i+1}. {feat['feature']}: {feat['importance']:.4f}")
            
            report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Evaluation report saved to {output_path}")
        
        return report_text