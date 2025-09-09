"""
Advanced feature engineering for earthquake prediction.
Creates temporal, spatial, geological and seismic sequence features.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config

class EarthquakeFeatureEngineer:
    """Feature engineering class for earthquake prediction."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set from raw earthquake data.
        
        Args:
            df: DataFrame with earthquake data
            
        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            return df
            
        # Make a copy to avoid modifying original data
        df_features = df.copy()
        
        # Temporal features (do this first, before removing time columns)
        df_features = self._create_temporal_features(df_features)
        
        # Spatial features  
        df_features = self._create_spatial_features(df_features)
        
        # Geological features (simulated)
        df_features = self._create_geological_features(df_features)
        
        # Seismic sequence features (do this before removing time columns)
        df_features = self._create_sequence_features(df_features)
        
        # Statistical features
        df_features = self._create_statistical_features(df_features)
        
        # Now remove datetime columns after all temporal processing is done
        datetime_cols = ['time', 'updated']
        for col in datetime_cols:
            if col in df_features.columns:
                df_features = df_features.drop(columns=[col])
        
        return df_features
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if 'time' not in df.columns:
            return df
            
        # Ensure time is datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Basic temporal features
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['day_of_year'] = df['time'].dt.dayofyear
        df['month'] = df['time'].dt.month
        df['quarter'] = df['time'].dt.quarter
        df['year'] = df['time'].dt.year
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Time since epoch (for trend analysis)
        df['time_since_epoch'] = (df['time'] - pd.Timestamp('1970-01-01')).dt.total_seconds()
        
        # Lunar cycle approximation (29.5 day cycle)
        lunar_cycle = 29.53  # days
        days_since_ref = (df['time'] - pd.Timestamp('2000-01-01')).dt.total_seconds() / 86400
        df['lunar_phase'] = (days_since_ref % lunar_cycle) / lunar_cycle
        df['lunar_sin'] = np.sin(2 * np.pi * df['lunar_phase'])
        df['lunar_cos'] = np.cos(2 * np.pi * df['lunar_phase'])
        
        return df
    
    def _create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spatial and geographical features."""
        if not all(col in df.columns for col in ['latitude', 'longitude']):
            return df
        
        # Distance from major tectonic features (simplified)
        major_faults = [
            {'name': 'San Andreas', 'lat': 37.0, 'lon': -122.0},
            {'name': 'Ring of Fire', 'lat': 0.0, 'lon': -120.0},  
            {'name': 'Japan Trench', 'lat': 38.0, 'lon': 142.0},
            {'name': 'Turkey Fault', 'lat': 40.0, 'lon': 30.0}
        ]
        
        for fault in major_faults:
            distance = self._haversine_distance(
                df['latitude'], df['longitude'],
                fault['lat'], fault['lon']
            )
            df[f'distance_to_{fault["name"].lower().replace(" ", "_")}'] = distance
        
        # Distance to nearest major fault
        fault_distances = [df[f'distance_to_{fault["name"].lower().replace(" ", "_")}'] for fault in major_faults]
        df['distance_to_nearest_fault'] = np.min(fault_distances, axis=0)
        
        # Depth-related features
        if 'depth' in df.columns:
            df['depth_log'] = np.log1p(np.abs(df['depth']))
            df['depth_squared'] = df['depth'] ** 2
            df['is_shallow'] = (df['depth'] < 70).astype(int)  # Shallow earthquakes
            df['is_intermediate'] = ((df['depth'] >= 70) & (df['depth'] < 300)).astype(int)
            df['is_deep'] = (df['depth'] >= 300).astype(int)
        
        # Regional clustering
        df = self._create_spatial_clusters(df)
        
        return df
    
    def _create_geological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create simulated geological features."""
        if not all(col in df.columns for col in ['latitude', 'longitude']):
            return df
            
        # Simulate rock type based on location (simplified)
        # In reality, this would come from geological databases
        df['rock_type'] = self._simulate_rock_type(df['latitude'], df['longitude'])
        
        # Simulate fault type based on region  
        df['fault_type'] = self._simulate_fault_type(df['latitude'], df['longitude'])
        
        # Simulate crustal thickness (affects earthquake characteristics)
        df['crustal_thickness'] = self._simulate_crustal_thickness(df['latitude'])
        
        # Tectonic setting
        df['tectonic_setting'] = self._simulate_tectonic_setting(df['latitude'], df['longitude'])
        
        return df
    
    def _create_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to earthquake sequences (foreshocks, aftershocks)."""
        if 'time' not in df.columns or len(df) < 2:
            return df
            
        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)
        
        # Time since previous earthquake  
        df['time_since_prev'] = (df['time'] - df['time'].shift(1)).dt.total_seconds() / 3600  # hours
        df['time_since_prev'] = df['time_since_prev'].fillna(df['time_since_prev'].median())
        
        # For rolling statistics, we need to be more careful with datetime indexing
        try:
            # Rolling statistics for sequence analysis
            windows = [3, 7, 30]  # 3, 7, 30 day windows
            
            for window in windows:
                # Simple count of events in time window (approximated)
                df[f'events_last_{window}d'] = df.groupby(['latitude', 'longitude'])['magnitude'].rolling(
                    window=min(window, len(df)), min_periods=1
                ).count().reset_index(level=[0,1], drop=True)
                
                # Average magnitude in window
                df[f'avg_mag_last_{window}d'] = df.groupby(['latitude', 'longitude'])['magnitude'].rolling(
                    window=min(window, len(df)), min_periods=1
                ).mean().reset_index(level=[0,1], drop=True)
                
                # Max magnitude in window
                df[f'max_mag_last_{window}d'] = df.groupby(['latitude', 'longitude'])['magnitude'].rolling(
                    window=min(window, len(df)), min_periods=1
                ).max().reset_index(level=[0,1], drop=True)
            
            # Fill NaN values with reasonable defaults
            sequence_cols = [col for col in df.columns if any(x in col for x in ['events_last', 'avg_mag', 'max_mag'])]
            for col in sequence_cols:
                df[col] = df[col].fillna(0)
                
        except Exception:
            # Fallback: create simple sequence features
            for window in [3, 7, 30]:
                df[f'events_last_{window}d'] = 1  # Default value
                df[f'avg_mag_last_{window}d'] = df['magnitude'] if 'magnitude' in df.columns else 5.0
                df[f'max_mag_last_{window}d'] = df['magnitude'] if 'magnitude' in df.columns else 5.0
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features from existing columns."""
        
        # Magnitude-related features
        if 'magnitude' in df.columns:
            df['magnitude_squared'] = df['magnitude'] ** 2
            df['magnitude_log'] = np.log1p(df['magnitude'])
            df['is_significant'] = (df['magnitude'] >= 5.0).astype(int)
            df['is_major'] = (df['magnitude'] >= 6.0).astype(int)
            df['is_great'] = (df['magnitude'] >= 7.0).astype(int)
        
        # Location encoding (for categorical algorithms)
        if 'place' in df.columns:
            # Simple hash-based encoding for location
            df['location_hash'] = df['place'].astype(str).apply(hash).abs() % 1000
        
        # Network and quality features
        quality_features = ['nst', 'gap', 'dmin', 'rms', 'magNst']
        for feature in quality_features:
            if feature in df.columns:
                df[f'{feature}_log'] = np.log1p(np.abs(df[feature].fillna(0)))
        
        return df
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points on Earth."""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _create_spatial_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spatial clusters using DBSCAN."""
        if len(df) < 5:  # Need minimum points for clustering
            df['spatial_cluster'] = 0
            return df
            
        try:
            # Use DBSCAN for spatial clustering  
            coords = df[['latitude', 'longitude']].values
            
            # Scale coordinates (roughly 1 degree = 111 km)
            coords_scaled = coords * [111, 111 * np.cos(np.radians(coords[:, 0]))]
            
            dbscan = DBSCAN(eps=50, min_samples=3)  # 50 km radius, min 3 points
            clusters = dbscan.fit_predict(coords_scaled)
            
            df['spatial_cluster'] = clusters
            df['is_clustered'] = (clusters != -1).astype(int)
            
        except Exception:
            # Fallback to simple grid-based clustering
            df['spatial_cluster'] = (
                (df['latitude'] // 2) * 1000 + (df['longitude'] // 2)
            ).astype(int)
            df['is_clustered'] = 1
            
        return df
    
    def _simulate_rock_type(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Simulate rock type based on location."""
        # Simplified geological mapping
        rock_types = ['sedimentary', 'igneous', 'metamorphic', 'volcanic']
        
        # Use lat/lon to determine rock type (simplified)
        type_index = ((np.abs(lat) + np.abs(lon)) * 10).astype(int) % len(rock_types)
        return np.array(rock_types)[type_index]
    
    def _simulate_fault_type(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Simulate fault type based on location."""
        fault_types = ['strike_slip', 'normal', 'reverse', 'thrust']
        
        # Use geographic features to simulate fault types
        type_index = ((lat * 3 + lon * 2) * 5).astype(int) % len(fault_types)
        return np.array(fault_types)[type_index]
    
    def _simulate_crustal_thickness(self, lat: np.ndarray) -> np.ndarray:
        """Simulate crustal thickness based on latitude."""
        # Oceanic crust: ~7km, Continental crust: ~35km
        # Simplified: thicker at higher latitudes (continental), thinner at equator (oceanic)
        base_thickness = 20 + 15 * np.cos(np.radians(lat * 2))  # 5-35 km range
        noise = np.random.normal(0, 3, len(lat))  # Add some noise
        return np.clip(base_thickness + noise, 5, 50)
    
    def _simulate_tectonic_setting(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Simulate tectonic setting based on location."""
        settings = ['subduction_zone', 'mid_ocean_ridge', 'transform_fault', 'intraplate']
        
        # Simplified tectonic mapping
        setting_index = ((np.abs(lat) + np.abs(lon/2)) * 7).astype(int) % len(settings)
        return np.array(settings)[setting_index]
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Return feature groups for analysis and interpretation."""
        return {
            'temporal': [
                'hour', 'day_of_week', 'month', 'quarter', 'hour_sin', 'hour_cos',
                'day_sin', 'day_cos', 'month_sin', 'month_cos', 'lunar_sin', 'lunar_cos'
            ],
            'spatial': [
                'latitude', 'longitude', 'depth', 'depth_log', 'depth_squared',
                'distance_to_nearest_fault', 'spatial_cluster', 'is_clustered'
            ],
            'geological': [
                'rock_type', 'fault_type', 'crustal_thickness', 'tectonic_setting',
                'is_shallow', 'is_intermediate', 'is_deep'
            ],
            'sequence': [
                'time_since_prev', 'events_last_3d', 'events_last_7d', 'events_last_30d',
                'avg_mag_last_3d', 'avg_mag_last_7d', 'avg_mag_last_30d'
            ],
            'magnitude': [
                'magnitude', 'magnitude_squared', 'magnitude_log', 
                'is_significant', 'is_major', 'is_great'
            ]
        }