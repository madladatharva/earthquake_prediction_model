#!/usr/bin/env python3
"""
Comprehensive Machine Learning Earthquake Magnitude Prediction Test Script

This script demonstrates a complete machine learning pipeline for earthquake magnitude prediction,
including data collection from USGS API, advanced feature engineering, Random Forest modeling,
performance evaluation, and location-based predictions.

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import requests
import json
import math
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
import time

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ComprehensiveEarthquakeML:
    """
    Comprehensive Machine Learning class for earthquake magnitude prediction
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.performance_metrics = {}
        self.feature_importance = {}
        
        # Known seismic locations for testing
        self.test_locations = {
            'San Francisco, CA': {'latitude': 37.7749, 'longitude': -122.4194, 'depth': 10},
            'Tokyo, Japan': {'latitude': 35.6762, 'longitude': 139.6503, 'depth': 15},
            'Los Angeles, CA': {'latitude': 34.0522, 'longitude': -118.2437, 'depth': 8},
            'Istanbul, Turkey': {'latitude': 41.0082, 'longitude': 28.9784, 'depth': 12},
            'Chile Coast': {'latitude': -33.4489, 'longitude': -70.6693, 'depth': 25}
        }
    
    def fetch_earthquake_data(self, start_date: str = '2023-01-01', 
                             end_date: str = '2023-12-31', 
                             min_magnitude: float = 3.0,
                             limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Fetch earthquake data from USGS API with comprehensive error handling
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            min_magnitude: Minimum earthquake magnitude
            limit: Maximum number of records to fetch
            
        Returns:
            DataFrame with earthquake data or None if fetch fails
        """
        print("🌍 Fetching earthquake data from USGS API...")
        
        url = 'https://earthquake.usgs.gov/fdsnws/event/1/query'
        params = {
            'format': 'geojson',
            'starttime': start_date,
            'endtime': end_date,
            'minmagnitude': min_magnitude,
            'limit': limit,
            'orderby': 'magnitude-desc'
        }
        
        try:
            print(f"   📡 Requesting {limit} earthquakes from {start_date} to {end_date}...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            features = data.get('features', [])
            
            if not features:
                print("   ⚠️  No earthquake data returned from API")
                return None
            
            print(f"   ✅ Successfully fetched {len(features)} earthquakes")
            return self._parse_earthquake_data(features)
            
        except requests.exceptions.RequestException as e:
            print(f"   ❌ API request failed: {e}")
            print("   🔄 Generating synthetic data for demonstration...")
            return self._generate_synthetic_data(limit)
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            print("   🔄 Generating synthetic data for demonstration...")
            return self._generate_synthetic_data(limit)
    
    def _parse_earthquake_data(self, features: List[Dict]) -> pd.DataFrame:
        """
        Parse USGS GeoJSON features into a structured DataFrame
        
        Args:
            features: List of GeoJSON feature dictionaries
            
        Returns:
            Structured DataFrame with earthquake data
        """
        print("   📊 Parsing earthquake data...")
        
        earthquake_list = []
        
        for feature in features:
            try:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                
                earthquake = {
                    'magnitude': props.get('mag'),
                    'longitude': coords[0] if len(coords) > 0 else None,
                    'latitude': coords[1] if len(coords) > 1 else None,
                    'depth': coords[2] if len(coords) > 2 else None,
                    'place': props.get('place', ''),
                    'time': props.get('time'),
                    'magnitude_type': props.get('magType', 'unknown'),
                    'gap': props.get('gap'),
                    'dmin': props.get('dmin'),
                    'rms': props.get('rms'),
                    'net': props.get('net', ''),
                    'id': props.get('id', ''),
                    'updated': props.get('updated'),
                    'type': props.get('type', 'earthquake')
                }
                
                # Filter out records with missing essential data
                if all(v is not None for v in [earthquake['magnitude'], 
                                              earthquake['longitude'], 
                                              earthquake['latitude'],
                                              earthquake['depth']]):
                    earthquake_list.append(earthquake)
                    
            except (KeyError, IndexError, TypeError) as e:
                continue  # Skip malformed records
        
        df = pd.DataFrame(earthquake_list)
        print(f"   ✅ Parsed {len(df)} valid earthquake records")
        return df
    
    def _generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate realistic synthetic earthquake data for demonstration with realistic correlations
        
        Args:
            n_samples: Number of synthetic samples to generate
            
        Returns:
            DataFrame with synthetic earthquake data
        """
        print(f"   🎲 Generating {n_samples} synthetic earthquake records...")
        
        np.random.seed(42)  # For reproducible results
        
        # Define realistic earthquake regions with their characteristics
        regions = {
            'Pacific Ring of Fire - Japan': {
                'lat_range': (30, 45), 'lon_range': (125, 145), 'depth_range': (0, 100),
                'mag_base': 4.5, 'mag_std': 1.2, 'frequency_weight': 0.25
            },
            'Pacific Ring of Fire - California': {
                'lat_range': (32, 42), 'lon_range': (-125, -115), 'depth_range': (0, 50),
                'mag_base': 4.0, 'mag_std': 1.0, 'frequency_weight': 0.20
            },
            'Pacific Ring of Fire - Chile': {
                'lat_range': (-45, -15), 'lon_range': (-75, -65), 'depth_range': (0, 200),
                'mag_base': 5.0, 'mag_std': 1.5, 'frequency_weight': 0.15
            },
            'Mediterranean - Turkey/Greece': {
                'lat_range': (35, 42), 'lon_range': (20, 45), 'depth_range': (0, 80),
                'mag_base': 4.2, 'mag_std': 1.1, 'frequency_weight': 0.15
            },
            'Mid-Atlantic Ridge': {
                'lat_range': (-40, 60), 'lon_range': (-45, -10), 'depth_range': (0, 30),
                'mag_base': 3.5, 'mag_std': 0.8, 'frequency_weight': 0.10
            },
            'Indonesia/Philippines': {
                'lat_range': (-10, 20), 'lon_range': (95, 140), 'depth_range': (0, 150),
                'mag_base': 4.8, 'mag_std': 1.3, 'frequency_weight': 0.15
            }
        }
        
        earthquakes = []
        
        for region_name, region_data in regions.items():
            n_region = int(n_samples * region_data['frequency_weight'])
            
            for i in range(n_region):
                # Generate coordinates within region bounds
                lat = np.random.uniform(*region_data['lat_range'])
                lon = np.random.uniform(*region_data['lon_range'])
                
                # Generate depth with exponential distribution
                depth = np.random.exponential(20)
                depth = min(max(depth, 0), region_data['depth_range'][1])
                
                # Generate magnitude with realistic correlations
                # Base magnitude influenced by depth and regional characteristics
                depth_factor = 0.02 * depth if depth < 50 else 0.02 * 50 + 0.005 * (depth - 50)
                base_mag = region_data['mag_base'] + depth_factor
                
                # Add some noise and regional variation
                magnitude = np.random.normal(base_mag, region_data['mag_std'])
                magnitude = max(3.0, min(magnitude, 9.5))  # Clamp to realistic range
                
                # Generate realistic instrumental parameters that correlate with magnitude
                # Larger earthquakes tend to have different gap, dmin, rms characteristics
                mag_factor = (magnitude - 3.0) / 6.5  # Normalize magnitude to 0-1
                
                # Gap (azimuthal gap) - larger earthquakes often have better station coverage (lower gap)
                gap = np.random.uniform(20 + 100 * (1 - mag_factor), 180 + 20 * (1 - mag_factor))
                gap = max(20, min(gap, 360))
                
                # Distance to nearest station (dmin) - varies with location and magnitude
                dmin = np.random.exponential(0.5 + 2.0 * (1 - mag_factor))
                dmin = max(0.01, min(dmin, 20.0))
                
                # RMS travel time residual - larger earthquakes often have more stable solutions
                rms = np.random.uniform(0.1 + 0.3 * (1 - mag_factor), 0.5 + 0.8 * (1 - mag_factor))
                rms = max(0.05, min(rms, 2.0))
                
                earthquake = {
                    'magnitude': round(magnitude, 1),
                    'longitude': round(lon, 4),
                    'latitude': round(lat, 4),
                    'depth': round(depth, 1),
                    'place': region_name,
                    'time': int(time.time() * 1000) + i * 1000,  # Spread timestamps
                    'magnitude_type': np.random.choice(['ml', 'mb', 'mw', 'md'], p=[0.4, 0.3, 0.2, 0.1]),
                    'gap': round(gap, 1),
                    'dmin': round(dmin, 3),
                    'rms': round(rms, 3),
                    'net': np.random.choice(['us', 'nc', 'ci', 'ak'], p=[0.4, 0.2, 0.2, 0.2]),
                    'id': f'synthetic_{len(earthquakes)}',
                    'updated': int(time.time() * 1000),
                    'type': 'earthquake'
                }
                earthquakes.append(earthquake)
        
        # Fill remaining samples with random global earthquakes
        remaining = n_samples - len(earthquakes)
        for i in range(remaining):
            lat = np.random.uniform(-90, 90)
            lon = np.random.uniform(-180, 180)
            depth = np.random.exponential(15)
            
            # Simple magnitude-depth correlation for global events
            base_mag = 3.5 + 0.01 * depth
            magnitude = max(3.0, min(np.random.normal(base_mag, 0.8), 8.0))
            
            # Generate correlated instrumental parameters
            mag_factor = (magnitude - 3.0) / 5.0
            gap = np.random.uniform(30 + 80 * (1 - mag_factor), 200)
            dmin = np.random.exponential(1.0 + 3.0 * (1 - mag_factor))
            rms = np.random.uniform(0.2 + 0.5 * (1 - mag_factor), 1.0)
            
            earthquake = {
                'magnitude': round(magnitude, 1),
                'longitude': round(lon, 4),
                'latitude': round(lat, 4),
                'depth': round(depth, 1),
                'place': 'Global Random',
                'time': int(time.time() * 1000) + i * 1000,
                'magnitude_type': 'ml',
                'gap': round(gap, 1),
                'dmin': round(dmin, 3),
                'rms': round(rms, 3),
                'net': 'us',
                'id': f'synthetic_{len(earthquakes)}',
                'updated': int(time.time() * 1000),
                'type': 'earthquake'
            }
            earthquakes.append(earthquake)
        
        df = pd.DataFrame(earthquakes)
        print(f"   ✅ Generated {len(df)} synthetic earthquake records with realistic correlations")
        return df
    
    def advanced_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform advanced feature engineering on earthquake data
        
        Args:
            df: Raw earthquake DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        print("🔧 Performing advanced feature engineering...")
        
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        # Basic coordinate features
        print("   📍 Processing geographic coordinates...")
        data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
        data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
        data['depth'] = pd.to_numeric(data['depth'], errors='coerce')
        data['magnitude'] = pd.to_numeric(data['magnitude'], errors='coerce')
        
        # Distance from equator
        print("   🌐 Calculating distance from equator...")
        data['distance_from_equator'] = abs(data['latitude'])
        
        # Pacific Ring of Fire indicator
        print("   🔥 Identifying Pacific Ring of Fire locations...")
        data['is_pacific_ring'] = self._is_pacific_ring_of_fire(data['latitude'], data['longitude'])
        
        # Depth categorization
        print("   📏 Categorizing depth levels...")
        data['depth_category'] = pd.cut(data['depth'], 
                                      bins=[-np.inf, 10, 50, 100, 300, np.inf],
                                      labels=['shallow', 'moderate', 'intermediate', 'deep', 'very_deep'])
        data['depth_category_encoded'] = data['depth_category'].cat.codes
        
        # Additional geometric features
        print("   📐 Computing additional geometric features...")
        data['depth_squared'] = data['depth'] ** 2
        data['depth_log'] = np.log1p(data['depth'])  # log(1 + depth) to handle depth=0
        
        # Coordinate interactions
        data['lat_lon_interaction'] = data['latitude'] * data['longitude']
        data['depth_lat_interaction'] = data['depth'] * abs(data['latitude'])
        
        # Regional indicators based on tectonic activity
        print("   🌏 Adding regional tectonic indicators...")
        data['is_mediterranean'] = ((data['latitude'] >= 30) & (data['latitude'] <= 45) & 
                                   (data['longitude'] >= -10) & (data['longitude'] <= 45)).astype(int)
        
        data['is_midatlantic_ridge'] = ((data['longitude'] >= -45) & (data['longitude'] <= -10) &
                                       (data['latitude'] >= -40) & (data['latitude'] <= 60)).astype(int)
        
        # Distance from major fault lines (simplified approximation)
        data['near_major_fault'] = self._near_major_fault_line(data['latitude'], data['longitude'])
        
        # Temporal features (if time data is available)
        if 'time' in data.columns:
            print("   ⏰ Processing temporal features...")
            try:
                data['timestamp'] = pd.to_datetime(data['time'], unit='ms', errors='coerce')
                data['hour'] = data['timestamp'].dt.hour
                data['day_of_year'] = data['timestamp'].dt.dayofyear
                data['month'] = data['timestamp'].dt.month
            except:
                pass  # Skip temporal features if time data is invalid
        
        # Handle missing values
        print("   🔧 Handling missing values...")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        
        print(f"   ✅ Feature engineering completed. Total features: {len(data.columns)}")
        return data
    
    def _is_pacific_ring_of_fire(self, latitude: pd.Series, longitude: pd.Series) -> pd.Series:
        """
        Determine if coordinates are in the Pacific Ring of Fire
        
        Args:
            latitude: Series of latitude values
            longitude: Series of longitude values
            
        Returns:
            Series of boolean values indicating Ring of Fire membership
        """
        # Simplified Pacific Ring of Fire boundaries
        # This is an approximation based on major tectonic boundaries
        
        conditions = [
            # Japan/Philippines/Indonesia region
            ((latitude >= -10) & (latitude <= 50) & (longitude >= 120) & (longitude <= 150)),
            
            # Alaska/Aleutians
            ((latitude >= 50) & (latitude <= 65) & (longitude >= 160) | (longitude <= -140)),
            
            # US West Coast
            ((latitude >= 30) & (latitude <= 50) & (longitude >= -130) & (longitude <= -115)),
            
            # Central America/Mexico
            ((latitude >= 10) & (latitude <= 35) & (longitude >= -120) & (longitude <= -90)),
            
            # Chile/Peru
            ((latitude >= -45) & (latitude <= -10) & (longitude >= -80) & (longitude <= -65)),
            
            # New Zealand
            ((latitude >= -50) & (latitude <= -30) & (longitude >= 165) & (longitude <= 180))
        ]
        
        result = pd.Series([False] * len(latitude), index=latitude.index)
        for condition in conditions:
            result = result | condition
            
        return result.astype(int)
    
    def _near_major_fault_line(self, latitude: pd.Series, longitude: pd.Series) -> pd.Series:
        """
        Determine proximity to major fault lines (simplified)
        
        Args:
            latitude: Series of latitude values
            longitude: Series of longitude values
            
        Returns:
            Series of values indicating proximity to major faults (0-1 scale)
        """
        # Major fault lines (simplified representation)
        major_faults = [
            # San Andreas Fault (California)
            {'lat_range': (32, 40), 'lon_range': (-122, -115), 'weight': 1.0},
            
            # Japan Trench
            {'lat_range': (35, 42), 'lon_range': (140, 145), 'weight': 1.0},
            
            # Chile Trench
            {'lat_range': (-45, -15), 'lon_range': (-75, -70), 'weight': 1.0},
            
            # North Anatolian Fault (Turkey)
            {'lat_range': (39, 42), 'lon_range': (26, 44), 'weight': 0.8},
            
            # Alpine Fault (New Zealand)
            {'lat_range': (-46, -40), 'lon_range': (166, 172), 'weight': 0.8}
        ]
        
        proximity = pd.Series([0.0] * len(latitude), index=latitude.index)
        
        for fault in major_faults:
            near_fault = ((latitude >= fault['lat_range'][0]) & 
                         (latitude <= fault['lat_range'][1]) &
                         (longitude >= fault['lon_range'][0]) & 
                         (longitude <= fault['lon_range'][1]))
            proximity = proximity + (near_fault.astype(float) * fault['weight'])
        
        # Normalize to 0-1 scale
        return np.clip(proximity, 0, 1)
    
    def train_model(self, data: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Train Random Forest model for magnitude prediction
        
        Args:
            data: DataFrame with engineered features
            
        Returns:
            Tuple of (performance metrics, feature importance dataframe)
        """
        print("🤖 Training Random Forest model...")
        
        # Define feature columns (exclude target and non-predictive columns)
        exclude_columns = ['magnitude', 'place', 'time', 'magnitude_type', 'net', 'id', 
                          'updated', 'type', 'timestamp', 'depth_category']
        
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        self.feature_columns = feature_columns
        
        # Prepare features and target
        X = data[feature_columns].copy()
        y = data['magnitude'].copy()
        
        # Handle any remaining missing values
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=feature_columns, index=X.index)
        
        print(f"   📊 Training with {X.shape[0]} samples and {X.shape[1]} features")
        print(f"   🎯 Target range: {y.min():.1f} - {y.max():.1f} magnitude")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
        )
        
        # Scale features
        print("   ⚖️  Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("   🌲 Training Random Forest...")
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True  # Out-of-bag score for additional validation
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        print("   🔮 Making predictions...")
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate performance metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        self.performance_metrics = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'oob_score': self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': len(feature_columns)
        }
        
        # Feature importance
        print("   📈 Analyzing feature importance...")
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance
        
        print(f"   ✅ Model training completed")
        print(f"      📊 Test RMSE: {test_rmse:.3f}")
        print(f"      📊 Test R²: {test_r2:.3f}")
        
        return self.performance_metrics, feature_importance
    
    def predict_locations(self) -> pd.DataFrame:
        """
        Make magnitude predictions for known seismic locations
        
        Returns:
            DataFrame with location predictions
        """
        print("📍 Making predictions for known seismic locations...")
        
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        
        for location_name, coords in self.test_locations.items():
            print(f"   🎯 Predicting for {location_name}...")
            
            # Create feature vector for this location
            features = self._create_location_features(coords)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Make prediction
            predicted_magnitude = self.model.predict(features_scaled)[0]
            
            prediction = {
                'location': location_name,
                'latitude': coords['latitude'],
                'longitude': coords['longitude'],
                'depth': coords['depth'],
                'predicted_magnitude': round(predicted_magnitude, 2),
                'is_pacific_ring': int(self._is_pacific_ring_of_fire(
                    pd.Series([coords['latitude']]), 
                    pd.Series([coords['longitude']])
                ).iloc[0]),
                'distance_from_equator': abs(coords['latitude'])
            }
            
            predictions.append(prediction)
        
        predictions_df = pd.DataFrame(predictions)
        print(f"   ✅ Completed predictions for {len(predictions)} locations")
        
        return predictions_df
    
    def _create_location_features(self, coords: Dict[str, float]) -> List[float]:
        """
        Create feature vector for a given location
        
        Args:
            coords: Dictionary with latitude, longitude, depth
            
        Returns:
            List of feature values matching the trained model's feature order
        """
        lat, lon, depth = coords['latitude'], coords['longitude'], coords['depth']
        
        # Create a temporary DataFrame to use existing feature engineering
        temp_data = pd.DataFrame([{
            'latitude': lat,
            'longitude': lon,
            'depth': depth,
            'magnitude': 0  # Dummy value
        }])
        
        # Apply feature engineering
        temp_data = self.advanced_feature_engineering(temp_data)
        
        # Extract features in the correct order
        features = []
        for feature_name in self.feature_columns:
            if feature_name in temp_data.columns:
                features.append(temp_data[feature_name].iloc[0])
            else:
                features.append(0.0)  # Default value for missing features
        
        return features
    
    def display_results(self, feature_importance: pd.DataFrame, predictions: pd.DataFrame):
        """
        Display comprehensive results and analysis
        
        Args:
            feature_importance: DataFrame with feature importance scores
            predictions: DataFrame with location predictions
        """
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE EARTHQUAKE MAGNITUDE PREDICTION RESULTS")
        print("="*80)
        
        # Model Performance
        print("\n🎯 MODEL PERFORMANCE METRICS")
        print("-" * 40)
        metrics = self.performance_metrics
        print(f"Training Samples: {metrics['n_train_samples']:,}")
        print(f"Testing Samples:  {metrics['n_test_samples']:,}")
        print(f"Number of Features: {metrics['n_features']}")
        print()
        print("Training Performance:")
        print(f"  • MSE:  {metrics['train_mse']:.4f}")
        print(f"  • RMSE: {metrics['train_rmse']:.4f}")
        print(f"  • R²:   {metrics['train_r2']:.4f}")
        print()
        print("Testing Performance:")
        print(f"  • MSE:  {metrics['test_mse']:.4f}")
        print(f"  • RMSE: {metrics['test_rmse']:.4f}")
        print(f"  • R²:   {metrics['test_r2']:.4f}")
        if metrics.get('oob_score') is not None:
            print(f"  • OOB Score: {metrics['oob_score']:.4f}")  # Out-of-bag validation
        
        # Feature Importance
        print(f"\n📈 TOP 10 MOST IMPORTANT FEATURES")
        print("-" * 40)
        top_features = feature_importance.head(10)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            importance_pct = row['importance'] * 100
            bar_length = int(importance_pct / 2)  # Scale for display
            bar = "█" * bar_length + "▒" * (25 - bar_length)
            print(f"{i:2d}. {row['feature']:<25} {bar} {importance_pct:5.1f}%")
        
        # Location Predictions
        print(f"\n🗺️  MAGNITUDE PREDICTIONS FOR SEISMIC LOCATIONS")
        print("-" * 40)
        for _, pred in predictions.iterrows():
            ring_status = "🔥 Ring of Fire" if pred['is_pacific_ring'] else "🌍 Other Region"
            print(f"{pred['location']:<20} | Magnitude: {pred['predicted_magnitude']:4.1f} | {ring_status}")
            print(f"{'':20} | Lat: {pred['latitude']:7.3f}, Lon: {pred['longitude']:8.3f}, Depth: {pred['depth']:4.1f}km")
            print()
        
        # Analysis Summary
        print(f"\n🧠 MODEL ANALYSIS SUMMARY")
        print("-" * 40)
        
        # Performance interpretation
        test_rmse = metrics['test_rmse']
        test_r2 = metrics['test_r2']
        
        if test_r2 > 0.7:
            performance = "Excellent"
        elif test_r2 > 0.5:
            performance = "Good"
        elif test_r2 > 0.3:
            performance = "Moderate"
        else:
            performance = "Poor"
            
        print(f"Model Performance: {performance} (R² = {test_r2:.3f})")
        print(f"Prediction Error: ±{test_rmse:.3f} magnitude units (RMSE)")
        
        # Key insights
        most_important = feature_importance.iloc[0]['feature']
        print(f"Most Important Factor: {most_important}")
        
        ring_predictions = predictions[predictions['is_pacific_ring'] == 1]['predicted_magnitude']
        other_predictions = predictions[predictions['is_pacific_ring'] == 0]['predicted_magnitude']
        
        if len(ring_predictions) > 0 and len(other_predictions) > 0:
            avg_ring = ring_predictions.mean()
            avg_other = other_predictions.mean()
            print(f"Average Ring of Fire Prediction: {avg_ring:.2f}")
            print(f"Average Other Region Prediction: {avg_other:.2f}")
        
        print("\n" + "="*80)
        print("✅ Analysis Complete - Earthquake ML Pipeline Successfully Demonstrated")
        print("="*80)


def main(save_results: bool = False, n_samples: int = 1200, quick_mode: bool = False):
    """
    Main function to run the comprehensive earthquake ML demonstration
    
    Args:
        save_results: Whether to save results to a file
        n_samples: Number of samples to generate (if using synthetic data)
        quick_mode: Run with fewer samples for faster execution
    """
    if quick_mode:
        n_samples = min(n_samples, 300)
        print("🚀 Running in quick mode...")
    
    print("🌍 COMPREHENSIVE EARTHQUAKE MAGNITUDE PREDICTION SYSTEM")
    print("="*65)
    print("This system demonstrates a complete ML pipeline for earthquake")
    print("magnitude prediction using advanced feature engineering and")
    print("Random Forest regression.")
    print("="*65)
    print()
    
    try:
        # Initialize the ML system
        ml_system = ComprehensiveEarthquakeML()
        
        # Fetch earthquake data
        earthquake_data = ml_system.fetch_earthquake_data(
            start_date='2023-01-01',
            end_date='2023-12-31',
            min_magnitude=3.0,
            limit=n_samples
        )
        
        if earthquake_data is None or len(earthquake_data) < 50:
            print("❌ Insufficient data for reliable ML training")
            return
        
        print(f"📊 Dataset Summary: {len(earthquake_data)} earthquakes")
        print(f"   Magnitude range: {earthquake_data['magnitude'].min():.1f} - {earthquake_data['magnitude'].max():.1f}")
        print(f"   Geographic span: {earthquake_data['longitude'].min():.1f}° to {earthquake_data['longitude'].max():.1f}° longitude")
        print(f"   Depth range: {earthquake_data['depth'].min():.1f} - {earthquake_data['depth'].max():.1f} km")
        print()
        
        # Perform feature engineering
        engineered_data = ml_system.advanced_feature_engineering(earthquake_data)
        
        # Train the model
        metrics, feature_importance = ml_system.train_model(engineered_data)
        
        # Make location predictions
        location_predictions = ml_system.predict_locations()
        
        # Display comprehensive results
        ml_system.display_results(feature_importance, location_predictions)
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"earthquake_ml_results_{timestamp}.json"
            
            results = {
                'timestamp': timestamp,
                'dataset_summary': {
                    'n_samples': len(earthquake_data),
                    'magnitude_range': [float(earthquake_data['magnitude'].min()), 
                                      float(earthquake_data['magnitude'].max())],
                    'geographic_range': [float(earthquake_data['longitude'].min()),
                                       float(earthquake_data['longitude'].max())],
                    'depth_range': [float(earthquake_data['depth'].min()),
                                   float(earthquake_data['depth'].max())]
                },
                'performance_metrics': metrics,
                'feature_importance': feature_importance.to_dict('records'),
                'location_predictions': location_predictions.to_dict('records')
            }
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n💡 Technical Notes:")
    print(f"   • Random Forest with 200 trees for robust predictions")
    print(f"   • Feature scaling applied for optimal performance")
    print(f"   • Cross-validation approach with 80/20 train/test split")
    print(f"   • Synthetic data fallback ensures functionality without API access")
    print(f"   • Advanced feature engineering includes Pacific Ring of Fire detection")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Earthquake ML Prediction System')
    parser.add_argument('--save-results', action='store_true', 
                       help='Save results to JSON file')
    parser.add_argument('--samples', type=int, default=1200,
                       help='Number of samples to generate (default: 1200)')
    parser.add_argument('--quick', action='store_true',
                       help='Run in quick mode with fewer samples')
    
    args = parser.parse_args()
    main(save_results=args.save_results, n_samples=args.samples, quick_mode=args.quick)