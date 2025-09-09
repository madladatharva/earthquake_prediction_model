"""
Data preprocessing pipeline for earthquake prediction.
Handles data cleaning, validation, and preparation for ML models.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config

class EarthquakeDataPreprocessor:
    """Data preprocessing pipeline for earthquake prediction models."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_columns = []
        self.target_column = 'magnitude'
        self.fitted = False
        self.logger = logging.getLogger(__name__)
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'magnitude') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit preprocessing pipeline and transform data.
        
        Args:
            df: Input DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            Tuple of (processed_features, target_series)
        """
        self.target_column = target_col
        
        # Split features and target
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df.copy()
            y = None
        
        # Store original feature columns
        self.feature_columns = X.columns.tolist()
        
        # Process features
        X_processed = self._fit_transform_features(X)
        
        self.fitted = True
        
        return X_processed, y
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Extract features (exclude target if present)
        if self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
        else:
            X = df.copy()
        
        # Ensure all expected columns are present
        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            self.logger.warning(f"Missing columns in transform: {missing_cols}")
            # Add missing columns with default values
            for col in missing_cols:
                X[col] = 0
        
        # Remove extra columns
        X = X[self.feature_columns]
        
        return self._transform_features(X)
    
    def _fit_transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform features."""
        X = X.copy()
        
        # Identify column types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle missing values
        if numeric_cols:
            self.imputers['numeric'] = SimpleImputer(strategy='median')
            X[numeric_cols] = self.imputers['numeric'].fit_transform(X[numeric_cols])
        
        if categorical_cols:
            self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = self.imputers['categorical'].fit_transform(X[categorical_cols])
        
        # Encode categorical variables
        X = self._encode_categorical_features(X, categorical_cols, fit=True)
        
        # Scale numerical features
        X = self._scale_numerical_features(X, fit=True)
        
        # Handle infinite and very large values
        X = self._handle_extreme_values(X)
        
        return X
    
    def _transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted preprocessors."""
        X = X.copy()
        
        # Identify column types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle missing values
        if numeric_cols and 'numeric' in self.imputers:
            X[numeric_cols] = self.imputers['numeric'].transform(X[numeric_cols])
        
        if categorical_cols and 'categorical' in self.imputers:
            X[categorical_cols] = self.imputers['categorical'].transform(X[categorical_cols])
        
        # Encode categorical variables
        X = self._encode_categorical_features(X, categorical_cols, fit=False)
        
        # Scale numerical features
        X = self._scale_numerical_features(X, fit=False)
        
        # Handle extreme values
        X = self._handle_extreme_values(X)
        
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame, categorical_cols: List[str], fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        X = X.copy()
        
        for col in categorical_cols:
            if col not in X.columns:
                continue
                
            if fit:
                # Use label encoding for high cardinality, one-hot for low cardinality
                n_unique = X[col].nunique()
                
                if n_unique <= 10:  # One-hot encoding
                    self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = self.encoders[col].fit_transform(X[[col]])
                    encoded_df = pd.DataFrame(
                        encoded, 
                        columns=[f"{col}_{cat}" for cat in self.encoders[col].categories_[0]],
                        index=X.index
                    )
                    X = X.drop(columns=[col])
                    X = pd.concat([X, encoded_df], axis=1)
                    
                else:  # Label encoding
                    self.encoders[col] = LabelEncoder()
                    X[col] = self.encoders[col].fit_transform(X[col].astype(str))
                    
            else:  # Transform only
                if col in self.encoders:
                    encoder = self.encoders[col]
                    
                    if isinstance(encoder, OneHotEncoder):
                        encoded = encoder.transform(X[[col]])
                        encoded_df = pd.DataFrame(
                            encoded,
                            columns=[f"{col}_{cat}" for cat in encoder.categories_[0]],
                            index=X.index
                        )
                        X = X.drop(columns=[col])
                        X = pd.concat([X, encoded_df], axis=1)
                        
                    else:  # Label encoder
                        # Handle unknown categories
                        unique_vals = set(X[col].astype(str))
                        known_vals = set(encoder.classes_)
                        unknown_vals = unique_vals - known_vals
                        
                        if unknown_vals:
                            X[col] = X[col].astype(str).apply(
                                lambda x: x if x in known_vals else encoder.classes_[0]
                            )
                        
                        X[col] = encoder.transform(X[col].astype(str))
        
        return X
    
    def _scale_numerical_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return X
        
        if fit:
            self.scalers['features'] = StandardScaler()
            X[numeric_cols] = self.scalers['features'].fit_transform(X[numeric_cols])
        else:
            if 'features' in self.scalers:
                X[numeric_cols] = self.scalers['features'].transform(X[numeric_cols])
        
        return X
    
    def _handle_extreme_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite and very large values."""
        X = X.copy()
        
        # Replace infinite values with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median for numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        
        # Clip extreme values to reasonable ranges
        for col in numeric_cols:
            if X[col].dtype in ['float64', 'float32']:
                q01, q99 = X[col].quantile([0.01, 0.99])
                X[col] = X[col].clip(lower=q01, upper=q99)
        
        return X
    
    def create_train_test_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        test_size: float = None,
        temporal_split: bool = True,
        time_column: str = 'time'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create train/test split with option for temporal splitting.
        
        Args:
            X: Features DataFrame
            y: Target series
            test_size: Size of test set
            temporal_split: Whether to use temporal split (more realistic for time series)
            time_column: Name of time column for temporal split
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = test_size or Config.TEST_SIZE
        
        if temporal_split and time_column in X.columns:
            # Sort by time and split chronologically
            sort_idx = X[time_column].sort_values().index
            X_sorted = X.loc[sort_idx]
            y_sorted = y.loc[sort_idx]
            
            split_idx = int(len(X_sorted) * (1 - test_size))
            
            X_train = X_sorted.iloc[:split_idx]
            X_test = X_sorted.iloc[split_idx:]
            y_train = y_sorted.iloc[:split_idx]
            y_test = y_sorted.iloc[split_idx:]
            
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=Config.RANDOM_STATE
            )
        
        self.logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def create_time_series_splits(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_splits: int = None
    ) -> TimeSeriesSplit:
        """
        Create time series cross-validation splits.
        
        Args:
            X: Features DataFrame  
            y: Target series
            n_splits: Number of CV splits
            
        Returns:
            TimeSeriesSplit object
        """
        n_splits = n_splits or Config.CV_FOLDS
        return TimeSeriesSplit(n_splits=n_splits)
    
    def get_preprocessing_info(self) -> Dict:
        """Get information about the preprocessing pipeline."""
        info = {
            'fitted': self.fitted,
            'target_column': self.target_column,
            'n_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'scalers': list(self.scalers.keys()),
            'encoders': list(self.encoders.keys()),
            'imputers': list(self.imputers.keys())
        }
        
        return info