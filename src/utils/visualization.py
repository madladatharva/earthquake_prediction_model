"""
Visualization utilities for earthquake prediction models and results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config

# Set style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

class EarthquakeVisualization:
    """Visualization utilities for earthquake prediction system."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 10)
    
    def plot_model_comparison(
        self,
        model_results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive comparison plot for multiple models.
        
        Args:
            model_results: Dictionary of model evaluation results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Filter out failed models
        valid_results = {
            name: results for name, results in model_results.items() 
            if 'error' not in results
        }
        
        if len(valid_results) < 2:
            raise ValueError("Need at least 2 valid models for comparison")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        models = list(valid_results.keys())
        
        # 1. Test Metrics Comparison (top-left)
        ax = axes[0, 0]
        metrics = ['r2', 'rmse', 'mae', 'mape']
        metric_data = {metric: [] for metric in metrics}
        
        for model in models:
            test_metrics = valid_results[model].get('test_metrics', {})
            for metric in metrics:
                metric_data[metric].append(test_metrics.get(metric, 0))
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            if metric == 'r2':
                ax.bar(x + i * width, metric_data[metric], width, 
                      label=metric.upper(), alpha=0.8)
            else:
                # Plot RMSE, MAE, MAPE on secondary axis (they have different scales)
                pass
        
        ax.set_xlabel('Models')
        ax.set_ylabel('R² Score')
        ax.set_title('Test Performance Metrics')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Cross-Validation Stability (top-right)
        ax = axes[0, 1]
        cv_means = []
        cv_stds = []
        
        for model in models:
            cv_results = valid_results[model].get('cv_results', {})
            r2_cv = cv_results.get('r2', {})
            cv_means.append(r2_cv.get('mean', 0))
            cv_stds.append(r2_cv.get('std', 0))
        
        bars = ax.bar(models, cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
        ax.set_ylabel('Cross-Validation R²')
        ax.set_title('Cross-Validation Performance')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val, std_val in zip(bars, cv_means, cv_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                   f'{mean_val:.3f}±{std_val:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        # 3. Error Metrics Comparison (bottom-left)
        ax = axes[1, 0]
        error_metrics = ['rmse', 'mae']
        
        x = np.arange(len(models))
        width = 0.35
        
        for i, metric in enumerate(error_metrics):
            values = [valid_results[model].get('test_metrics', {}).get(metric, 0) 
                     for model in models]
            ax.bar(x + i * width, values, width, label=metric.upper(), alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Error Value')
        ax.set_title('Error Metrics Comparison')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Performance Categories (bottom-right)
        ax = axes[1, 1]
        categories = []
        colors_map = {'excellent': 'green', 'good': 'blue', 'fair': 'orange', 'poor': 'red'}
        
        for model in models:
            summary = valid_results[model].get('performance_summary', {})
            category = summary.get('performance_category', 'unknown')
            categories.append(category)
        
        # Count categories
        category_counts = pd.Series(categories).value_counts()
        colors = [colors_map.get(cat, 'gray') for cat in category_counts.index]
        
        wedges, texts, autotexts = ax.pie(
            category_counts.values, 
            labels=category_counts.index, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax.set_title('Performance Categories Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot prediction results with actual vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            uncertainties: Prediction uncertainties (optional)
            model_name: Name of the model
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{model_name} - Prediction Analysis', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted scatter plot
        ax = axes[0]
        
        if uncertainties is not None:
            scatter = ax.scatter(y_true, y_pred, c=uncertainties, 
                              cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax, label='Prediction Uncertainty')
        else:
            ax.scatter(y_true, y_pred, alpha=0.6, s=50, color=self.colors[0])
        
        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Magnitude')
        ax.set_ylabel('Predicted Magnitude')
        ax.set_title('Actual vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Residuals plot
        ax = axes[1]
        residuals = y_true - y_pred
        
        ax.scatter(y_pred, residuals, alpha=0.6, s=50, color=self.colors[1])
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        
        # Add trend line
        try:
            z = np.polyfit(y_pred, residuals, 1)
            p = np.poly1d(z)
            ax.plot(y_pred, p(y_pred), "b-", alpha=0.8, label='Trend')
            ax.legend()
        except:
            pass
        
        ax.set_xlabel('Predicted Magnitude')
        ax.set_ylabel('Residuals (Actual - Predicted)')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)
        
        # 3. Residuals distribution
        ax = axes[2]
        
        ax.hist(residuals, bins=20, alpha=0.7, color=self.colors[2], edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.8, label='Zero Error')
        
        # Add normal distribution overlay
        try:
            from scipy.stats import norm
            mu, sigma = norm.fit(residuals)
            x = np.linspace(residuals.min(), residuals.max(), 100)
            y = norm.pdf(x, mu, sigma)
            
            # Scale to histogram
            y = y * len(residuals) * (residuals.max() - residuals.min()) / 20
            
            ax.plot(x, y, 'g-', alpha=0.8, label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
        except ImportError:
            pass
        
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_importance: pd.DataFrame,
        top_n: int = 15,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_importance: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to display
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Get top N features
        top_features = feature_importance.head(top_n)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                      color=self.colors[:len(top_features)])
        
        # Customize plot
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Most Important Features')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importance:.4f}', ha='left', va='center', fontsize=9)
        
        # Invert y-axis to have highest importance at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_earthquake_map(
        self,
        earthquake_data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot earthquake locations on a world map.
        
        Args:
            earthquake_data: DataFrame with 'latitude', 'longitude', 'magnitude' columns
            predictions: Optional predicted magnitudes
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 8)) if predictions is not None else plt.subplots(1, 1, figsize=(15, 8))
        
        if predictions is not None:
            # Plot actual magnitudes
            ax = axes[0]
        else:
            ax = axes if not hasattr(axes, '__len__') else axes[0]
        
        scatter = ax.scatter(
            earthquake_data['longitude'], 
            earthquake_data['latitude'],
            c=earthquake_data['magnitude'],
            s=earthquake_data['magnitude'] * 10,  # Size proportional to magnitude
            cmap='Reds',
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Actual Earthquake Magnitudes')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Magnitude')
        
        # World map boundaries (simplified)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        
        # Add basic continental outlines (very simplified)
        self._add_continental_outlines(ax)
        
        if predictions is not None:
            # Plot predicted magnitudes
            ax = axes[1]
            
            scatter = ax.scatter(
                earthquake_data['longitude'], 
                earthquake_data['latitude'],
                c=predictions,
                s=predictions * 10,
                cmap='Blues',
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('Predicted Earthquake Magnitudes')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Predicted Magnitude')
            
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            
            self._add_continental_outlines(ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_temporal_patterns(
        self,
        earthquake_data: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot temporal patterns in earthquake data.
        
        Args:
            earthquake_data: DataFrame with 'time' and 'magnitude' columns
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Earthquake Patterns', fontsize=16, fontweight='bold')
        
        # Ensure time column is datetime
        if 'time' in earthquake_data.columns:
            earthquake_data = earthquake_data.copy()
            earthquake_data['time'] = pd.to_datetime(earthquake_data['time'])
        else:
            raise ValueError("DataFrame must contain 'time' column")
        
        # 1. Time series of magnitudes
        ax = axes[0, 0]
        ax.scatter(earthquake_data['time'], earthquake_data['magnitude'], 
                  alpha=0.6, s=30, color=self.colors[0])
        ax.set_xlabel('Time')
        ax.set_ylabel('Magnitude')
        ax.set_title('Earthquake Magnitudes Over Time')
        ax.grid(True, alpha=0.3)
        
        # 2. Magnitude distribution by hour
        ax = axes[0, 1]
        earthquake_data['hour'] = earthquake_data['time'].dt.hour
        hourly_mags = earthquake_data.groupby('hour')['magnitude'].mean()
        
        ax.bar(hourly_mags.index, hourly_mags.values, alpha=0.7, color=self.colors[1])
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Magnitude')
        ax.set_title('Average Magnitude by Hour')
        ax.grid(True, alpha=0.3)
        
        # 3. Magnitude distribution by day of week
        ax = axes[1, 0]
        earthquake_data['day_of_week'] = earthquake_data['time'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_mags = earthquake_data.groupby('day_of_week')['magnitude'].mean().reindex(day_order)
        
        ax.bar(range(len(daily_mags)), daily_mags.values, alpha=0.7, color=self.colors[2])
        ax.set_xticks(range(len(daily_mags)))
        ax.set_xticklabels([day[:3] for day in day_order])
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Average Magnitude')
        ax.set_title('Average Magnitude by Day of Week')
        ax.grid(True, alpha=0.3)
        
        # 4. Monthly earthquake frequency
        ax = axes[1, 1]
        earthquake_data['month'] = earthquake_data['time'].dt.month_name()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_counts = earthquake_data.groupby('month').size().reindex(month_order)
        
        ax.bar(range(len(monthly_counts)), monthly_counts.values, alpha=0.7, color=self.colors[3])
        ax.set_xticks(range(len(monthly_counts)))
        ax.set_xticklabels([month[:3] for month in month_order])
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Earthquakes')
        ax.set_title('Earthquake Frequency by Month')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _add_continental_outlines(self, ax):
        """Add very basic continental outlines to map."""
        # This is a very simplified representation
        # In a production system, you'd use proper map data
        
        # Major landmasses (very simplified rectangles)
        continents = [
            # North America
            {'x': [-140, -140, -60, -60, -140], 'y': [20, 70, 70, 20, 20]},
            # South America
            {'x': [-80, -80, -35, -35, -80], 'y': [-55, 15, 15, -55, -55]},
            # Europe
            {'x': [-10, -10, 40, 40, -10], 'y': [35, 70, 70, 35, 35]},
            # Africa
            {'x': [-20, -20, 50, 50, -20], 'y': [-35, 35, 35, -35, -35]},
            # Asia
            {'x': [25, 25, 180, 180, 25], 'y': [10, 75, 75, 10, 10]},
            # Australia
            {'x': [110, 110, 155, 155, 110], 'y': [-45, -10, -10, -45, -45]}
        ]
        
        for continent in continents:
            ax.plot(continent['x'], continent['y'], 'k-', alpha=0.3, linewidth=1)
    
    def create_dashboard(
        self,
        model_results: Dict[str, Dict[str, Any]],
        earthquake_data: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            model_results: Model evaluation results
            earthquake_data: Raw earthquake data
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Earthquake Prediction System Dashboard', fontsize=20, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Model Performance Summary (top row, span 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        
        valid_results = {k: v for k, v in model_results.items() if 'error' not in v}
        if valid_results:
            models = list(valid_results.keys())
            r2_scores = [valid_results[m]['test_metrics'].get('r2', 0) for m in models]
            
            bars = ax1.bar(models, r2_scores, alpha=0.7, color=self.colors[:len(models)])
            ax1.set_ylabel('R² Score')
            ax1.set_title('Model Performance Comparison')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars, r2_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
        
        # 2. Best Model Details (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        
        if valid_results:
            best_model = max(valid_results.keys(), 
                           key=lambda k: valid_results[k]['test_metrics'].get('r2', 0))
            best_score = valid_results[best_model]['test_metrics']['r2']
            
            ax2.text(0.5, 0.7, f'Best Model:', ha='center', va='center', 
                    fontsize=14, fontweight='bold', transform=ax2.transAxes)
            ax2.text(0.5, 0.5, best_model, ha='center', va='center', 
                    fontsize=16, color=self.colors[0], transform=ax2.transAxes)
            ax2.text(0.5, 0.3, f'R² = {best_score:.4f}', ha='center', va='center', 
                    fontsize=14, transform=ax2.transAxes)
            
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
        
        # 3. Earthquake Map (middle row, span 2 columns)
        ax3 = fig.add_subplot(gs[1, :2])
        
        if 'latitude' in earthquake_data.columns and 'longitude' in earthquake_data.columns:
            scatter = ax3.scatter(
                earthquake_data['longitude'], 
                earthquake_data['latitude'],
                c=earthquake_data['magnitude'],
                s=earthquake_data['magnitude'] * 5,
                cmap='Reds',
                alpha=0.6
            )
            ax3.set_xlabel('Longitude')
            ax3.set_ylabel('Latitude')
            ax3.set_title('Earthquake Locations and Magnitudes')
            ax3.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax3, shrink=0.6)
            cbar.set_label('Magnitude')
        
        # 4. Data Statistics (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        
        stats_text = f"""
Dataset Statistics:

Total Earthquakes: {len(earthquake_data)}

Magnitude Range:
  Min: {earthquake_data['magnitude'].min():.1f}
  Max: {earthquake_data['magnitude'].max():.1f}
  Mean: {earthquake_data['magnitude'].mean():.2f}

Features: {len(earthquake_data.columns)}

Models Evaluated: {len(model_results)}
Successful: {len(valid_results)}
        """
        
        ax4.text(0.1, 0.5, stats_text, ha='left', va='center', 
                fontsize=10, transform=ax4.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        # 5. Magnitude Distribution (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        
        ax5.hist(earthquake_data['magnitude'], bins=20, alpha=0.7, 
                color=self.colors[0], edgecolor='black')
        ax5.set_xlabel('Magnitude')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Magnitude Distribution')
        ax5.grid(True, alpha=0.3)
        
        # 6. Temporal Pattern (bottom middle)
        ax6 = fig.add_subplot(gs[2, 1])
        
        if 'time' in earthquake_data.columns:
            eq_data_temp = earthquake_data.copy()
            eq_data_temp['time'] = pd.to_datetime(eq_data_temp['time'])
            eq_data_temp['hour'] = eq_data_temp['time'].dt.hour
            
            hourly_counts = eq_data_temp.groupby('hour').size()
            ax6.bar(hourly_counts.index, hourly_counts.values, 
                   alpha=0.7, color=self.colors[1])
            ax6.set_xlabel('Hour of Day')
            ax6.set_ylabel('Count')
            ax6.set_title('Earthquakes by Hour')
            ax6.grid(True, alpha=0.3)
        
        # 7. Model Status Summary (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])
        
        # Performance categories
        categories = []
        for name, results in model_results.items():
            if 'error' not in results:
                summary = results.get('performance_summary', {})
                category = summary.get('performance_category', 'unknown')
                categories.append(category)
        
        if categories:
            category_counts = pd.Series(categories).value_counts()
            colors_map = {'excellent': 'green', 'good': 'blue', 'fair': 'orange', 'poor': 'red'}
            colors = [colors_map.get(cat, 'gray') for cat in category_counts.index]
            
            ax7.pie(category_counts.values, labels=category_counts.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
            ax7.set_title('Performance Categories')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_all_plots(
        self,
        model_results: Dict[str, Dict[str, Any]],
        earthquake_data: pd.DataFrame,
        output_dir: str = "./results"
    ):
        """
        Generate and save all visualization plots.
        
        Args:
            model_results: Model evaluation results
            earthquake_data: Raw earthquake data
            output_dir: Directory to save plots
        """
        from pathlib import Path
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        plots_to_generate = [
            ('model_comparison.png', lambda: self.plot_model_comparison(model_results)),
            ('earthquake_map.png', lambda: self.plot_earthquake_map(earthquake_data)),
            ('temporal_patterns.png', lambda: self.plot_temporal_patterns(earthquake_data)),
            ('dashboard.png', lambda: self.create_dashboard(model_results, earthquake_data))
        ]
        
        for filename, plot_func in plots_to_generate:
            try:
                filepath = Path(output_dir) / filename
                fig = plot_func()
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)  # Free memory
                print(f"Saved: {filepath}")
            except Exception as e:
                print(f"Failed to generate {filename}: {e}")
        
        print(f"All plots saved to {output_dir}/")