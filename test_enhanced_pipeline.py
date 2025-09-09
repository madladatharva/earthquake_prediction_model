"""
Test script for enhanced data pipeline.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.data_fetcher import EnhancedUSGSCollector
from data.feature_engineer import EarthquakeFeatureEngineer
from data.data_preprocessor import EarthquakeDataPreprocessor

def test_enhanced_pipeline():
    """Test the enhanced data pipeline."""
    print("🧪 Testing Enhanced Earthquake Data Pipeline")
    print("=" * 50)
    
    # 1. Test data fetching
    print("\n1. Testing Data Fetching...")
    collector = EnhancedUSGSCollector()
    
    try:
        # Fetch recent earthquake data
        df_raw = collector.fetch_enhanced_data(days_back=7, min_magnitude=4.0)
        print(f"   ✅ Fetched {len(df_raw)} earthquake records")
        print(f"   📊 Columns: {list(df_raw.columns[:5])}... (showing first 5)")
        
        if len(df_raw) > 0:
            print(f"   📈 Magnitude range: {df_raw['magnitude'].min():.1f} - {df_raw['magnitude'].max():.1f}")
            print(f"   🌍 Location range: lat({df_raw['latitude'].min():.1f}, {df_raw['latitude'].max():.1f})")
    except Exception as e:
        print(f"   ❌ Data fetching error: {e}")
        return False
    
    # 2. Test feature engineering
    print("\n2. Testing Feature Engineering...")
    engineer = EarthquakeFeatureEngineer()
    
    try:
        df_features = engineer.create_all_features(df_raw)
        print(f"   ✅ Created {len(df_features.columns)} features from {len(df_raw.columns)} raw columns")
        
        # Show feature groups
        feature_groups = engineer.get_feature_importance_groups()
        for group, features in feature_groups.items():
            available_features = [f for f in features if f in df_features.columns]
            print(f"   📊 {group.capitalize()}: {len(available_features)} features")
            
    except Exception as e:
        print(f"   ❌ Feature engineering error: {e}")
        return False
    
    # 3. Test data preprocessing
    print("\n3. Testing Data Preprocessing...")
    preprocessor = EarthquakeDataPreprocessor()
    
    try:
        X, y = preprocessor.fit_transform(df_features, target_col='magnitude')
        print(f"   ✅ Processed data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Test train/test split
        X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(X, y)
        print(f"   📊 Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"   🎯 Target range: {y.min():.1f} - {y.max():.1f}")
        
        # Get preprocessing info
        info = preprocessor.get_preprocessing_info()
        print(f"   🔧 Preprocessing info: {info['n_features']} features, {len(info['encoders'])} encoders")
        
    except Exception as e:
        print(f"   ❌ Preprocessing error: {e}")
        return False
    
    print("\n✅ All pipeline tests completed successfully!")
    print(f"📈 Ready for ML model training with {X_train.shape[0]} training samples")
    
    return True

if __name__ == "__main__":
    success = test_enhanced_pipeline()
    if success:
        print("\n🎉 Enhanced pipeline is ready for model training!")
    else:
        print("\n❌ Pipeline tests failed. Check the errors above.")