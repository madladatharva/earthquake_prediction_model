"""
Main prediction engine for the earthquake prediction system.
Integrates all components for real-time prediction with confidence intervals.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))

from data.data_fetcher import EnhancedUSGSCollector
from data.feature_engineer import EarthquakeFeatureEngineer
from data.data_preprocessor import EarthquakeDataPreprocessor
from models.ensemble_model import EarthquakeEnsembleModel
from utils.config import Config
from utils.model_evaluator import EarthquakeModelEvaluator
from utils.visualization import EarthquakeVisualization

class EarthquakePredictionEngine:
    """
    Main prediction engine that integrates all components for earthquake prediction.
    Provides real-time predictions with confidence intervals and alert systems.
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize the prediction engine.
        
        Args:
            config_override: Optional configuration overrides
        """
        self.config = Config()
        if config_override:
            for key, value in config_override.items():
                setattr(self.config, key, value)
        
        # Initialize components
        self.data_collector = EnhancedUSGSCollector()
        self.feature_engineer = EarthquakeFeatureEngineer()
        self.data_preprocessor = EarthquakeDataPreprocessor()
        self.ensemble_model = EarthquakeEnsembleModel()
        self.evaluator = EarthquakeModelEvaluator()
        self.visualizer = EarthquakeVisualization()
        
        # State variables
        self.is_trained = False
        self.last_training_time = None
        self.training_results = {}
        self.alert_thresholds = Config.ALERT_THRESHOLDS
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Ensure directories exist
        Config.ensure_directories()
    
    def train_system(
        self,
        days_back: int = 365,
        min_magnitude: float = 4.0,
        retrain: bool = False,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train the complete prediction system.
        
        Args:
            days_back: Number of days of historical data to use
            min_magnitude: Minimum magnitude threshold
            retrain: Whether to retrain if already trained
            save_model: Whether to save the trained model
            
        Returns:
            Training results summary
        """
        self.logger.info("Starting system training...")
        
        if self.is_trained and not retrain:
            self.logger.info("System already trained. Use retrain=True to force retraining.")
            return self.training_results
        
        try:
            # 1. Collect and prepare data
            self.logger.info("Collecting earthquake data...")
            raw_data = self.data_collector.fetch_enhanced_data(
                days_back=days_back,
                min_magnitude=min_magnitude
            )
            
            if len(raw_data) < 50:
                raise ValueError(f"Insufficient data: only {len(raw_data)} records found")
            
            self.logger.info(f"Collected {len(raw_data)} earthquake records")
            
            # 2. Feature engineering
            self.logger.info("Engineering features...")
            feature_data = self.feature_engineer.create_all_features(raw_data)
            
            # 3. Data preprocessing
            self.logger.info("Preprocessing data...")
            X, y = self.data_preprocessor.fit_transform(feature_data, target_col='magnitude')
            
            # 4. Train/test split with temporal awareness
            X_train, X_test, y_train, y_test = self.data_preprocessor.create_train_test_split(
                X, y, temporal_split=False  # Using random split for now
            )
            
            self.logger.info(f"Training data: {len(X_train)} samples, {X_train.shape[1]} features")
            
            # 5. Train ensemble model
            self.logger.info("Training ensemble model...")
            self.ensemble_model.initialize_models(['random_forest', 'xgboost', 'svm'])  # Skip neural_network for speed
            
            training_results = self.ensemble_model.train(
                X_train, y_train,
                X_val=X_test, y_val=y_test,
                ensemble_type='weighted_average',
                optimize_weights=True
            )
            
            # 6. Evaluate system performance
            self.logger.info("Evaluating system performance...")
            evaluation_results = self.evaluator.evaluate_model(
                self.ensemble_model, X_train, X_test, y_train, y_test,
                model_name="EnsemblePredictionSystem"
            )
            
            # 7. Save model if requested
            if save_model:
                model_path = Config.RESULTS_DIR / f"earthquake_prediction_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                self.ensemble_model.save_ensemble(str(model_path))
                self.logger.info(f"Model saved to {model_path}")
            
            # 8. Store results
            self.training_results = {
                'training_completed': datetime.now().isoformat(),
                'data_summary': {
                    'total_records': len(raw_data),
                    'features': X_train.shape[1],
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'magnitude_range': [float(y.min()), float(y.max())]
                },
                'model_performance': training_results,
                'evaluation_results': evaluation_results,
                'feature_groups': self.feature_engineer.get_feature_importance_groups()
            }
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            self.logger.info(f"Training completed successfully. R² = {training_results['ensemble_metrics']['r2']:.4f}")
            
            return self.training_results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def predict_single(
        self,
        earthquake_data: Union[pd.DataFrame, Dict],
        include_uncertainty: bool = True
    ) -> Dict[str, Any]:
        """
        Make a prediction for a single earthquake or location.
        
        Args:
            earthquake_data: Earthquake data as DataFrame row or dict
            include_uncertainty: Whether to include uncertainty estimation
            
        Returns:
            Prediction results with confidence intervals
        """
        if not self.is_trained:
            raise ValueError("System must be trained before making predictions")
        
        try:
            # Convert to DataFrame if dict
            if isinstance(earthquake_data, dict):
                data_df = pd.DataFrame([earthquake_data])
            else:
                data_df = earthquake_data.copy()
            
            # Feature engineering
            feature_data = self.feature_engineer.create_all_features(data_df)
            
            # Preprocessing (transform only, don't refit)
            X = self.data_preprocessor.transform(feature_data)
            
            # Make prediction
            if include_uncertainty:
                prediction, uncertainty = self.ensemble_model.predict_with_uncertainty(X)
            else:
                prediction = self.ensemble_model.predict(X)
                uncertainty = np.array([0.1])  # Default uncertainty
            
            # Generate alert level
            alert_level = self._calculate_alert_level(prediction[0], uncertainty[0])
            
            result = {
                'predicted_magnitude': float(prediction[0]),
                'uncertainty': float(uncertainty[0]),
                'confidence_interval_95%': [
                    float(prediction[0] - 1.96 * uncertainty[0]),
                    float(prediction[0] + 1.96 * uncertainty[0])
                ],
                'alert_level': alert_level,
                'prediction_time': datetime.now().isoformat(),
                'model_info': {
                    'ensemble_weights': self.ensemble_model.ensemble_weights,
                    'active_models': list(self.ensemble_model.models.keys())
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(
        self,
        earthquake_data: pd.DataFrame,
        include_uncertainty: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions for a batch of earthquakes.
        
        Args:
            earthquake_data: DataFrame with earthquake data
            include_uncertainty: Whether to include uncertainty estimation
            
        Returns:
            DataFrame with predictions and metadata
        """
        if not self.is_trained:
            raise ValueError("System must be trained before making predictions")
        
        try:
            # Feature engineering
            feature_data = self.feature_engineer.create_all_features(earthquake_data)
            
            # Preprocessing
            X = self.data_preprocessor.transform(feature_data)
            
            # Make predictions
            if include_uncertainty:
                predictions, uncertainties = self.ensemble_model.predict_with_uncertainty(X)
            else:
                predictions = self.ensemble_model.predict(X)
                uncertainties = np.full_like(predictions, 0.1)
            
            # Create results DataFrame
            results_df = earthquake_data.copy()
            results_df['predicted_magnitude'] = predictions
            results_df['uncertainty'] = uncertainties
            
            # Confidence intervals
            results_df['ci_lower_95%'] = predictions - 1.96 * uncertainties
            results_df['ci_upper_95%'] = predictions + 1.96 * uncertainties
            
            # Alert levels
            results_df['alert_level'] = [
                self._calculate_alert_level(pred, unc) 
                for pred, unc in zip(predictions, uncertainties)
            ]
            
            results_df['prediction_time'] = datetime.now().isoformat()
            
            return results_df
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            raise
    
    def get_real_time_predictions(
        self,
        region: Optional[Dict[str, float]] = None,
        min_magnitude: float = 4.0,
        hours_back: int = 24
    ) -> Dict[str, Any]:
        """
        Get real-time predictions for recent earthquake activity.
        
        Args:
            region: Geographic region bounds (minlat, maxlat, minlon, maxlon)
            min_magnitude: Minimum magnitude threshold
            hours_back: Hours of recent data to analyze
            
        Returns:
            Real-time prediction results
        """
        if not self.is_trained:
            raise ValueError("System must be trained before making real-time predictions")
        
        try:
            # Fetch recent data
            days_back = max(1, hours_back / 24)
            recent_data = self.data_collector.fetch_enhanced_data(
                days_back=days_back,
                min_magnitude=min_magnitude,
                region=region
            )
            
            if len(recent_data) == 0:
                return {
                    'status': 'no_recent_activity',
                    'message': 'No recent earthquake activity found',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Make predictions
            predictions_df = self.predict_batch(recent_data, include_uncertainty=True)
            
            # Analyze results
            high_risk_events = predictions_df[
                predictions_df['alert_level'].isin(['high', 'critical'])
            ]
            
            # Summary statistics
            summary = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'analysis_period': f'{hours_back} hours',
                'total_events': len(predictions_df),
                'high_risk_events': len(high_risk_events),
                'prediction_summary': {
                    'mean_magnitude': float(predictions_df['predicted_magnitude'].mean()),
                    'max_magnitude': float(predictions_df['predicted_magnitude'].max()),
                    'mean_uncertainty': float(predictions_df['uncertainty'].mean()),
                    'alert_distribution': predictions_df['alert_level'].value_counts().to_dict()
                },
                'high_risk_locations': self._identify_high_risk_regions(high_risk_events) if len(high_risk_events) > 0 else [],
                'predictions': predictions_df.to_dict('records')
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Real-time prediction failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def monitor_system_health(self) -> Dict[str, Any]:
        """
        Monitor the health and performance of the prediction system.
        
        Returns:
            System health report
        """
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'operational' if self.is_trained else 'not_trained',
            'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
            'model_info': {}
        }
        
        if self.is_trained:
            # Check model performance
            if self.training_results:
                performance = self.training_results.get('model_performance', {})
                ensemble_metrics = performance.get('ensemble_metrics', {})
                
                health_report['model_info'] = {
                    'r2_score': ensemble_metrics.get('r2', 0),
                    'rmse': ensemble_metrics.get('rmse', 0),
                    'active_models': list(self.ensemble_model.models.keys()) if hasattr(self.ensemble_model, 'models') else [],
                    'ensemble_weights': self.ensemble_model.ensemble_weights if hasattr(self.ensemble_model, 'ensemble_weights') else {}
                }
                
                # Health status based on performance
                r2_score = ensemble_metrics.get('r2', 0)
                if r2_score >= 0.8:
                    health_report['performance_status'] = 'excellent'
                elif r2_score >= 0.6:
                    health_report['performance_status'] = 'good'
                else:
                    health_report['performance_status'] = 'poor'
        
        # Check data freshness
        try:
            recent_data = self.data_collector.fetch_enhanced_data(days_back=1, min_magnitude=4.0)
            health_report['data_status'] = {
                'recent_events_24h': len(recent_data),
                'data_connection': 'active'
            }
        except Exception as e:
            health_report['data_status'] = {
                'recent_events_24h': 0,
                'data_connection': 'error',
                'error': str(e)
            }
        
        return health_report
    
    def create_prediction_report(
        self,
        predictions_df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> str:
        """
        Create a comprehensive prediction report.
        
        Args:
            predictions_df: DataFrame with predictions
            output_path: Optional path to save the report
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("# Earthquake Prediction Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Predictions: {len(predictions_df)}")
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("## Prediction Summary")
        report_lines.append(f"Mean Predicted Magnitude: {predictions_df['predicted_magnitude'].mean():.2f}")
        report_lines.append(f"Maximum Predicted Magnitude: {predictions_df['predicted_magnitude'].max():.2f}")
        report_lines.append(f"Mean Uncertainty: {predictions_df['uncertainty'].mean():.3f}")
        report_lines.append("")
        
        # Alert distribution
        alert_dist = predictions_df['alert_level'].value_counts()
        report_lines.append("## Alert Level Distribution")
        for level, count in alert_dist.items():
            percentage = (count / len(predictions_df)) * 100
            report_lines.append(f"- {level.title()}: {count} ({percentage:.1f}%)")
        report_lines.append("")
        
        # High-risk events
        high_risk = predictions_df[predictions_df['alert_level'].isin(['high', 'critical'])]
        if len(high_risk) > 0:
            report_lines.append("## High-Risk Events")
            report_lines.append(f"Number of high-risk events: {len(high_risk)}")
            
            for idx, event in high_risk.head(10).iterrows():
                report_lines.append(f"- Magnitude: {event['predicted_magnitude']:.2f} ± {event['uncertainty']:.3f}")
                if 'place' in event:
                    report_lines.append(f"  Location: {event['place']}")
                if 'latitude' in event and 'longitude' in event:
                    report_lines.append(f"  Coordinates: {event['latitude']:.2f}, {event['longitude']:.2f}")
        
        # Model information
        if hasattr(self, 'training_results') and self.training_results:
            report_lines.append("\n## Model Information")
            model_perf = self.training_results.get('model_performance', {})
            if 'ensemble_weights' in model_perf:
                report_lines.append("Model Weights:")
                for model, weight in model_perf['ensemble_weights'].items():
                    report_lines.append(f"- {model}: {weight:.3f}")
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to {output_path}")
        
        return report_text
    
    def _calculate_alert_level(self, prediction: float, uncertainty: float) -> str:
        """Calculate alert level based on prediction and uncertainty."""
        # High magnitude with low uncertainty = high alert
        # Moderate magnitude with high uncertainty = medium alert
        
        if prediction >= self.alert_thresholds['high_magnitude'] and uncertainty < 0.5:
            return 'critical'
        elif prediction >= self.alert_thresholds['high_magnitude'] or uncertainty > 1.0:
            return 'high'
        elif prediction >= 5.0 or uncertainty > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _identify_high_risk_regions(self, high_risk_events: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify geographic regions with high earthquake risk."""
        if 'latitude' not in high_risk_events.columns or 'longitude' not in high_risk_events.columns:
            return []
        
        regions = []
        
        # Simple clustering approach - in production, use proper spatial clustering
        for idx, event in high_risk_events.iterrows():
            region = {
                'latitude': float(event['latitude']),
                'longitude': float(event['longitude']),
                'predicted_magnitude': float(event['predicted_magnitude']),
                'uncertainty': float(event['uncertainty']),
                'alert_level': event['alert_level']
            }
            
            if 'place' in event:
                region['location'] = event['place']
            
            regions.append(region)
        
        return regions
    
    def save_system_state(self, filepath: str) -> None:
        """Save the current system state."""
        state = {
            'is_trained': self.is_trained,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_results': self.training_results,
            'alert_thresholds': self.alert_thresholds,
            'config': {
                'random_state': self.config.RANDOM_STATE,
                'ensemble_weights': self.config.ENSEMBLE_WEIGHTS
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"System state saved to {filepath}")
    
    def load_system_state(self, filepath: str) -> None:
        """Load a previously saved system state."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.is_trained = state.get('is_trained', False)
        
        if state.get('last_training_time'):
            self.last_training_time = datetime.fromisoformat(state['last_training_time'])
        
        self.training_results = state.get('training_results', {})
        self.alert_thresholds = state.get('alert_thresholds', Config.ALERT_THRESHOLDS)
        
        self.logger.info(f"System state loaded from {filepath}")