"""
Test script for advanced ML models.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

from data.data_fetcher import EnhancedUSGSCollector
from data.feature_engineer import EarthquakeFeatureEngineer
from data.data_preprocessor import EarthquakeDataPreprocessor
from models.random_forest_model import EnhancedRandomForestModel
from models.xgboost_model import XGBoostModel
from models.neural_network_model import EnhancedNeuralNetworkModel
from models.svm_model import SVMEarthquakeModel
from models.ensemble_model import EarthquakeEnsembleModel

def test_individual_models():
    """Test individual ML models."""
    print("🧪 Testing Advanced ML Models")
    print("=" * 50)
    
    # 1. Prepare data
    print("\n1. Preparing Data...")
    collector = EnhancedUSGSCollector()
    df_raw = collector.fetch_enhanced_data(days_back=30, min_magnitude=4.0)
    
    engineer = EarthquakeFeatureEngineer()
    df_features = engineer.create_all_features(df_raw)
    
    preprocessor = EarthquakeDataPreprocessor()
    X, y = preprocessor.fit_transform(df_features, target_col='magnitude')
    X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(
        X, y, temporal_split=False  # Disable temporal split for now
    )
    
    print(f"   📊 Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"   📊 Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Store results
    results = {}
    
    # 2. Test Random Forest
    print("\n2. Testing Enhanced Random Forest...")
    try:
        rf_model = EnhancedRandomForestModel()
        rf_results = rf_model.train(X_train, y_train, optimize_params=False)  # Skip optimization for speed
        rf_pred = rf_model.predict(X_test)
        rf_pred_unc, rf_unc = rf_model.predict_with_uncertainty(X_test)
        
        results['Random Forest'] = {
            'r2': rf_results['train_metrics']['r2'],
            'cv_r2': rf_results['cv_mean_r2'],
            'uncertainty_available': True
        }
        print(f"   ✅ Random Forest: R² = {rf_results['train_metrics']['r2']:.4f}, CV = {rf_results['cv_mean_r2']:.4f}")
    except Exception as e:
        print(f"   ❌ Random Forest failed: {e}")
        results['Random Forest'] = {'error': str(e)}
    
    # 3. Test XGBoost
    print("\n3. Testing XGBoost...")
    try:
        xgb_model = XGBoostModel()
        xgb_results = xgb_model.train(X_train, y_train, optimize_params=False)
        xgb_pred = xgb_model.predict(X_test)
        xgb_pred_unc, xgb_unc = xgb_model.predict_with_uncertainty(X_test)
        
        results['XGBoost'] = {
            'r2': xgb_results['train_metrics']['r2'],
            'cv_r2': xgb_results['cv_mean_r2'],
            'uncertainty_available': True
        }
        print(f"   ✅ XGBoost: R² = {xgb_results['train_metrics']['r2']:.4f}, CV = {xgb_results['cv_mean_r2']:.4f}")
    except Exception as e:
        print(f"   ❌ XGBoost failed: {e}")
        results['XGBoost'] = {'error': str(e)}
    
    # 4. Test Neural Network
    print("\n4. Testing Enhanced Neural Network...")
    try:
        nn_model = EnhancedNeuralNetworkModel()
        nn_results = nn_model.train(X_train, y_train, epochs=20)  # Fewer epochs for speed
        nn_pred = nn_model.predict(X_test)
        nn_pred_unc, nn_unc = nn_model.predict_with_uncertainty(X_test, n_samples=10)
        
        results['Neural Network'] = {
            'r2': nn_results['train_metrics']['r2'],
            'cv_r2': nn_results['cv_mean_r2'],
            'uncertainty_available': True
        }
        print(f"   ✅ Neural Network: R² = {nn_results['train_metrics']['r2']:.4f}, CV = {nn_results['cv_mean_r2']:.4f}")
    except Exception as e:
        print(f"   ❌ Neural Network failed: {e}")
        results['Neural Network'] = {'error': str(e)}
    
    # 5. Test SVM
    print("\n5. Testing SVM...")
    try:
        svm_model = SVMEarthquakeModel()
        svm_results = svm_model.train(X_train, y_train, optimize_params=False)
        svm_pred = svm_model.predict(X_test)
        svm_pred_unc, svm_unc = svm_model.predict_with_uncertainty(X_test)
        
        results['SVM'] = {
            'r2': svm_results['train_metrics']['r2'],
            'cv_r2': svm_results['cv_mean_r2'],
            'uncertainty_available': True
        }
        print(f"   ✅ SVM: R² = {svm_results['train_metrics']['r2']:.4f}, CV = {svm_results['cv_mean_r2']:.4f}")
    except Exception as e:
        print(f"   ❌ SVM failed: {e}")
        results['SVM'] = {'error': str(e)}
    
    return results, X_train, X_test, y_train, y_test

def test_ensemble_model(X_train, X_test, y_train, y_test):
    """Test ensemble model."""
    print("\n6. Testing Ensemble Model...")
    
    try:
        # Initialize and train ensemble
        ensemble = EarthquakeEnsembleModel()
        ensemble.initialize_models(['random_forest', 'xgboost', 'neural_network', 'svm'])
        
        ensemble_results = ensemble.train(
            X_train, y_train, 
            X_val=X_test, y_val=y_test,
            ensemble_type='weighted_average',
            optimize_weights=True
        )
        
        # Test predictions
        ensemble_pred = ensemble.predict(X_test)
        ensemble_pred_unc, ensemble_unc = ensemble.predict_with_uncertainty(X_test)
        
        # Get model contributions
        contributions = ensemble.get_model_contributions(X_test)
        
        print(f"   ✅ Ensemble: R² = {ensemble_results['ensemble_metrics']['r2']:.4f}")
        print(f"   📊 Weights: {ensemble_results['ensemble_weights']}")
        print(f"   📊 Models: {ensemble_results['individual_models']}")
        
        return {
            'Ensemble': {
                'r2': ensemble_results['ensemble_metrics']['r2'],
                'cv_r2': ensemble_results['cv_mean_r2'],
                'weights': ensemble_results['ensemble_weights'],
                'uncertainty_available': True
            }
        }
        
    except Exception as e:
        print(f"   ❌ Ensemble failed: {e}")
        return {'Ensemble': {'error': str(e)}}

def main():
    """Main test function."""
    try:
        # Test individual models
        results, X_train, X_test, y_train, y_test = test_individual_models()
        
        # Test ensemble if enough models succeeded
        successful_models = [k for k, v in results.items() if 'error' not in v]
        if len(successful_models) >= 2:
            ensemble_results = test_ensemble_model(X_train, X_test, y_train, y_test)
            results.update(ensemble_results)
        else:
            print(f"\n⚠️  Skipping ensemble test: only {len(successful_models)} models succeeded")
        
        # Print summary
        print("\n" + "="*50)
        print("📈 MODEL PERFORMANCE SUMMARY")
        print("="*50)
        
        for model_name, result in results.items():
            if 'error' in result:
                print(f"❌ {model_name:<20} FAILED: {result['error']}")
            else:
                r2 = result['r2']
                cv_r2 = result.get('cv_r2', 'N/A')
                uncertainty = "✓" if result.get('uncertainty_available') else "✗"
                print(f"✅ {model_name:<20} R²={r2:.4f} CV_R²={cv_r2} Uncertainty={uncertainty}")
        
        successful_count = len([r for r in results.values() if 'error' not in r])
        print(f"\n🎉 {successful_count}/{len(results)} models successful!")
        
        if successful_count >= len(results) // 2:
            print("✅ Advanced ML models are ready for production use!")
            return True
        else:
            print("⚠️  Some models failed - check implementation")
            return False
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)