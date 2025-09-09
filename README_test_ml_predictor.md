# Comprehensive Earthquake ML Predictor Test Script

## Overview

`test_ml_predictor.py` is a comprehensive machine learning test script that demonstrates earthquake magnitude prediction capabilities using advanced feature engineering and Random Forest regression.

## Features

### 🌍 **Data Collection**
- **USGS API Integration**: Fetches real earthquake data from USGS (1000+ earthquakes from 2023)
- **Fallback Mechanism**: Generates realistic synthetic data when API is unavailable
- **Robust Error Handling**: Graceful degradation with network connectivity issues

### 🔧 **Advanced Feature Engineering**
- **Geographic Features**: Longitude, latitude, depth processing
- **Distance Calculations**: Distance from equator
- **Tectonic Classification**: Pacific Ring of Fire zone detection
- **Depth Categorization**: Shallow, moderate, intermediate, deep, very deep
- **Interactive Features**: Coordinate interactions, depth transformations
- **Regional Indicators**: Mediterranean, Mid-Atlantic Ridge detection
- **Temporal Features**: Hour, day of year, month extraction

### 🤖 **Machine Learning Implementation**
- **Algorithm**: Random Forest Regressor with 200 trees
- **Data Splitting**: 80/20 train/test split with stratification
- **Feature Scaling**: StandardScaler for optimal performance
- **Performance Metrics**: MSE, RMSE, R², OOB Score
- **Cross-Validation**: Out-of-bag validation for additional robustness

### 📊 **Analysis & Evaluation**
- **Feature Importance**: Ranked analysis of predictive factors
- **Performance Visualization**: Easy-to-read performance bars
- **Location Predictions**: Tests on 5 known seismic locations:
  - San Francisco, CA
  - Tokyo, Japan
  - Los Angeles, CA
  - Istanbul, Turkey
  - Chile Coast

### 💡 **User-Friendly Features**
- **Progress Indicators**: Clear progress reporting throughout execution
- **Comprehensive Results**: Detailed performance metrics and analysis
- **Command Line Options**: Flexible execution modes
- **Result Export**: JSON export functionality for further analysis

## Usage

### Basic Usage
```bash
python test_ml_predictor.py
```

### Command Line Options
```bash
# Quick mode (fewer samples for faster execution)
python test_ml_predictor.py --quick

# Custom sample size
python test_ml_predictor.py --samples 500

# Save results to JSON file
python test_ml_predictor.py --save-results

# Combined options
python test_ml_predictor.py --quick --save-results --samples 300

# Help
python test_ml_predictor.py --help
```

### As a Python Module
```python
from test_ml_predictor import ComprehensiveEarthquakeML

# Initialize the system
ml_system = ComprehensiveEarthquakeML()

# Fetch and process data
data = ml_system.fetch_earthquake_data()
engineered_data = ml_system.advanced_feature_engineering(data)

# Train model
metrics, feature_importance = ml_system.train_model(engineered_data)

# Make predictions
predictions = ml_system.predict_locations()
```

## Output

The script provides comprehensive output including:

1. **Dataset Summary**: Number of earthquakes, magnitude/geographic/depth ranges
2. **Model Performance**: Training/testing MSE, RMSE, R² scores
3. **Feature Importance**: Top 10 most important predictive features
4. **Location Predictions**: Magnitude predictions for known seismic locations
5. **Analysis Summary**: Model performance interpretation and key insights

## Technical Details

### Performance Metrics
- **R² Score**: Coefficient of determination (higher = better)
- **RMSE**: Root Mean Square Error in magnitude units
- **OOB Score**: Out-of-bag validation score

### Feature Engineering
The script creates 19+ features from raw earthquake data:
- Basic coordinates and depth
- Distance from equator
- Pacific Ring of Fire indicator
- Depth categorization and transformations
- Geographic region indicators
- Coordinate interactions
- Temporal features (when available)

### Synthetic Data
When real API data is unavailable, the script generates realistic synthetic data that:
- Mimics real earthquake distributions by region
- Includes realistic correlations between magnitude, depth, and location
- Incorporates instrumental parameters (gap, dmin, rms) that correlate with magnitude
- Provides sufficient variation for meaningful ML training

## Dependencies

All dependencies are listed in `requirements.txt`:
- pandas, numpy: Data manipulation
- scikit-learn: Machine learning
- requests: API communication
- Standard Python libraries for additional functionality

## Limitations

- **Educational Purpose**: This is for demonstration and research, not operational earthquake prediction
- **Synthetic Data**: When API is unavailable, uses synthetic data that may not reflect all real-world complexities
- **Model Scope**: Focuses on magnitude prediction rather than location or timing prediction

## Example Output

```
🎯 MODEL PERFORMANCE METRICS
Training Samples: 960
Testing Samples:  240
Testing Performance:
  • RMSE: 0.967
  • R²:   0.424

📈 TOP FEATURES
1. gap               ████████████ 33.2%
2. rms               ████████▒▒▒▒ 26.0%
3. dmin              ███▒▒▒▒▒▒▒▒▒  7.3%

🗺️ LOCATION PREDICTIONS
San Francisco, CA | Magnitude: 7.2 | 🔥 Ring of Fire
Tokyo, Japan      | Magnitude: 7.5 | 🔥 Ring of Fire
```

This script serves as a comprehensive demonstration of the complete machine learning pipeline for earthquake data analysis and prediction.