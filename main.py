#!/usr/bin/env python3
"""
Main entry point for the Earthquake Prediction System.
Provides command-line interface for training, prediction, and system monitoring.
"""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from prediction_engine import EarthquakePredictionEngine
from utils.config import Config
from utils.visualization import EarthquakeVisualization

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('earthquake_prediction.log')
        ]
    )

def train_command(args):
    """Handle the train command."""
    print("🔧 Training Earthquake Prediction System...")
    print("=" * 50)
    
    engine = EarthquakePredictionEngine()
    
    try:
        results = engine.train_system(
            days_back=args.days_back,
            min_magnitude=args.min_magnitude,
            retrain=args.retrain,
            save_model=True
        )
        
        # Print results summary
        print("\n✅ Training completed successfully!")
        print(f"📊 Dataset: {results['data_summary']['total_records']} records")
        print(f"📈 Features: {results['data_summary']['features']}")
        print(f"🎯 Performance: R² = {results['model_performance']['ensemble_metrics']['r2']:.4f}")
        print(f"🤖 Models: {', '.join(results['model_performance']['individual_models'])}")
        
        # Save detailed results
        results_path = Path("results") / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📄 Detailed results saved to: {results_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def predict_command(args):
    """Handle the predict command."""
    print("🔮 Making Earthquake Predictions...")
    print("=" * 50)
    
    engine = EarthquakePredictionEngine()
    
    try:
        # Check if system is trained
        if not engine.is_trained:
            print("⚠️  System not trained. Training first...")
            train_result = engine.train_system(days_back=180, min_magnitude=4.0)
            if not train_result:
                print("❌ Training failed. Cannot make predictions.")
                return False
        
        # Get predictions
        if args.real_time:
            print("📡 Fetching real-time predictions...")
            results = engine.get_real_time_predictions(
                hours_back=args.hours_back,
                min_magnitude=args.min_magnitude
            )
            
            if results['status'] == 'success':
                print(f"\n📊 Analysis Summary:")
                print(f"   Total events: {results['total_events']}")
                print(f"   High-risk events: {results['high_risk_events']}")
                print(f"   Mean predicted magnitude: {results['prediction_summary']['mean_magnitude']:.2f}")
                print(f"   Max predicted magnitude: {results['prediction_summary']['max_magnitude']:.2f}")
                
                # Alert distribution
                print(f"\n🚨 Alert Distribution:")
                for level, count in results['prediction_summary']['alert_distribution'].items():
                    print(f"   {level.title()}: {count}")
                
                # High-risk locations
                if results['high_risk_locations']:
                    print(f"\n⚠️  High-Risk Locations:")
                    for location in results['high_risk_locations'][:5]:
                        print(f"   Magnitude: {location['predicted_magnitude']:.2f} ± {location['uncertainty']:.3f}")
                        if 'location' in location:
                            print(f"   Location: {location['location']}")
                
            else:
                print(f"⚠️  {results['message']}")
            
            # Save results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"📄 Results saved to: {args.output}")
        
        else:
            # Single prediction mode would go here
            print("📍 Single prediction mode not yet implemented in CLI")
            print("   Use --real-time for real-time predictions")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

def monitor_command(args):
    """Handle the monitor command."""
    print("📊 System Health Monitoring...")
    print("=" * 50)
    
    engine = EarthquakePredictionEngine()
    
    try:
        health = engine.monitor_system_health()
        
        print(f"\n🔧 System Status: {health['system_status'].upper()}")
        
        if health['last_training']:
            print(f"📅 Last Training: {health['last_training']}")
        
        if 'model_info' in health and health['model_info']:
            print(f"\n🤖 Model Performance:")
            print(f"   R² Score: {health['model_info']['r2_score']:.4f}")
            print(f"   RMSE: {health['model_info']['rmse']:.4f}")
            print(f"   Active Models: {', '.join(health['model_info']['active_models'])}")
            
            if health['model_info']['ensemble_weights']:
                print(f"   Model Weights:")
                for model, weight in health['model_info']['ensemble_weights'].items():
                    print(f"     {model}: {weight:.3f}")
        
        print(f"\n📡 Data Status:")
        data_status = health.get('data_status', {})
        print(f"   Recent Events (24h): {data_status.get('recent_events_24h', 'Unknown')}")
        print(f"   Connection: {data_status.get('data_connection', 'Unknown')}")
        
        if 'performance_status' in health:
            status_emoji = {'excellent': '🟢', 'good': '🟡', 'poor': '🔴'}
            emoji = status_emoji.get(health['performance_status'], '⚪')
            print(f"\n{emoji} Overall Performance: {health['performance_status'].upper()}")
        
        # Save health report
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(health, f, indent=2, default=str)
            print(f"\n📄 Health report saved to: {args.output}")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        return False

def visualize_command(args):
    """Handle the visualize command."""
    print("📈 Generating Visualizations...")
    print("=" * 50)
    
    engine = EarthquakePredictionEngine()
    
    try:
        # Check if system is trained
        if not engine.is_trained:
            print("⚠️  System not trained. Training first...")
            train_result = engine.train_system(days_back=180, min_magnitude=4.0)
            if not train_result:
                print("❌ Training failed. Cannot generate visualizations.")
                return False
        
        # Get some data for visualization
        recent_data = engine.data_collector.fetch_enhanced_data(days_back=30, min_magnitude=4.0)
        
        if len(recent_data) == 0:
            print("⚠️  No recent data available for visualization")
            return False
        
        # Make predictions for visualization
        predictions_df = engine.predict_batch(recent_data)
        
        # Generate visualizations
        visualizer = EarthquakeVisualization()
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"📊 Generating plots in {output_dir}...")
        
        # Earthquake map
        map_fig = visualizer.plot_earthquake_map(
            recent_data,
            predictions=predictions_df['predicted_magnitude'].values
        )
        map_path = output_dir / 'earthquake_map.png'
        map_fig.savefig(map_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Earthquake map: {map_path}")
        
        # Temporal patterns
        temporal_fig = visualizer.plot_temporal_patterns(recent_data)
        temporal_path = output_dir / 'temporal_patterns.png'
        temporal_fig.savefig(temporal_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Temporal patterns: {temporal_path}")
        
        # Prediction results
        pred_fig = visualizer.plot_prediction_results(
            recent_data['magnitude'].values,
            predictions_df['predicted_magnitude'].values,
            predictions_df['uncertainty'].values,
            model_name="Ensemble Model"
        )
        pred_path = output_dir / 'prediction_results.png'
        pred_fig.savefig(pred_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Prediction results: {pred_path}")
        
        # Dashboard
        if hasattr(engine, 'training_results') and engine.training_results:
            model_results = {'Ensemble': engine.evaluator.evaluation_results.get('EnsemblePredictionSystem', {})}
            if model_results['Ensemble']:
                dashboard_fig = visualizer.create_dashboard(model_results, recent_data)
                dashboard_path = output_dir / 'dashboard.png'
                dashboard_fig.savefig(dashboard_path, dpi=300, bbox_inches='tight')
                print(f"   ✅ Dashboard: {dashboard_path}")
        
        print(f"\n🎨 All visualizations saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Earthquake Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the system
  python main.py train --days-back 365 --min-magnitude 4.0
  
  # Get real-time predictions
  python main.py predict --real-time --hours-back 24
  
  # Monitor system health
  python main.py monitor
  
  # Generate visualizations
  python main.py visualize --output-dir ./plots
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the prediction system')
    train_parser.add_argument('--days-back', type=int, default=365, help='Days of historical data (default: 365)')
    train_parser.add_argument('--min-magnitude', type=float, default=4.0, help='Minimum magnitude threshold (default: 4.0)')
    train_parser.add_argument('--retrain', action='store_true', help='Force retraining if already trained')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--real-time', action='store_true', help='Get real-time predictions')
    predict_parser.add_argument('--hours-back', type=int, default=24, help='Hours of recent data for real-time mode (default: 24)')
    predict_parser.add_argument('--min-magnitude', type=float, default=4.0, help='Minimum magnitude threshold (default: 4.0)')
    predict_parser.add_argument('--output', '-o', help='Output file for results')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor system health')
    monitor_parser.add_argument('--output', '-o', help='Output file for health report')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    viz_parser.add_argument('--output-dir', default='./plots', help='Output directory for plots (default: ./plots)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command handler
    success = False
    
    if args.command == 'train':
        success = train_command(args)
    elif args.command == 'predict':
        success = predict_command(args)
    elif args.command == 'monitor':
        success = monitor_command(args)
    elif args.command == 'visualize':
        success = visualize_command(args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())