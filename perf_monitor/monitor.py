import pandas as pd
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
import whylogs as why
from whylogs.api.writer.whylabs import WhyLabsWriter
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)

# Load environment variables
load_dotenv()

class WhyLabsMonitor:
    def __init__(self, data_path='monitoring_data.csv'):
        self.data_path = data_path
        self.training_metrics = {
            'accuracy': 0.5646,
            'recall': 0.9355,
            'precision': 0.1162,
            'f1': 0.2068,
            'roc_auc': 0.8475
        }

        # Initialize WhyLabs
        self.whylabs_writer = self._init_whylabs()

    def _init_whylabs(self):
        """Initialize WhyLabs writer if configured"""
        if all(os.getenv(k) for k in ['WHYLABS_API_KEY', 'WHYLABS_ORG_ID', 'WHYLABS_DATASET_ID']):
            return WhyLabsWriter(
                api_key=os.getenv("WHYLABS_API_KEY"),
                org_id=os.getenv("WHYLABS_ORG_ID"),
                dataset_id=os.getenv("WHYLABS_DATASET_ID")
            )
        print("WhyLabs credentials not configured - skipping WhyLabs integration")
        return None

    def load_data(self):
        """Load monitoring data from CSV"""
        try:
            df = pd.read_csv(self.data_path)
            required_cols = ['age', 'avg_glucose_level', 'bmi', 
                            'hypertension', 'heart_disease', 
                            'predicted_y', 'true_y']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None

    def log_to_whylabs(self, df):
        """Log data to WhyLabs"""
        if self.whylabs_writer is None:
            return False

        try:
            # Create profile
            profile = why.log(df).profile()
            
            # Add performance metrics
            metrics = self.calculate_performance(df)
            profile.add_metadata("performance_metrics", json.dumps(metrics))
            
            # Write to WhyLabs
            self.whylabs_writer.write(profile)
            return True
        except Exception as e:
            print(f"Error logging to WhyLabs: {str(e)}")
            return False

    def calculate_performance(self, df):
        """Calculate performance metrics"""
        return {
            'accuracy': accuracy_score(df['true_y'], df['predicted_y']),
            'recall': recall_score(df['true_y'], df['predicted_y']),
            'precision': precision_score(df['true_y'], df['predicted_y']),
            'f1': f1_score(df['true_y'], df['predicted_y']),
            'roc_auc': roc_auc_score(df['true_y'], df['predicted_y']),
            'sample_size': len(df)
        }

    def generate_report(self):
        """Generate monitoring report"""
        df = self.load_data()
        if df is None:
            return {'status': 'error', 'message': 'Failed to load data'}
            
        # Log to WhyLabs
        whylabs_success = self.log_to_whylabs(df)
        
        # Calculate performance
        current_metrics = self.calculate_performance(df)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'whylabs_logged': whylabs_success,
            'performance': current_metrics,
            'alerts': self._check_anomalies(current_metrics)
        }
        
        return report

    def _check_anomalies(self, current_metrics):
        """Check for performance anomalies"""
        alerts = []
        for metric, value in current_metrics.items():
            if metric in self.training_metrics:
                drop = (self.training_metrics[metric] - value) / self.training_metrics[metric]
                if drop > 0.15:  # 15% performance drop
                    alerts.append(
                        f"Performance drop in {metric}: {drop:.1%} "
                        f"(from {self.training_metrics[metric]:.3f} to {value:.3f})"
                    )
        return alerts

    def save_report(self, report):
        """Save report to JSON file"""
        os.makedirs('reports', exist_ok=True)
        filename = f"reports/monitor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {filename}")
        return filename

if __name__ == "__main__":
    print("Starting WhyLabs Monitoring...")
    
    monitor = WhyLabsMonitor(data_path='monitoring_data.csv')
    report = monitor.generate_report()
    monitor.save_report(report)
    
    print("\n=== Monitoring Summary ===")
    print(f"Status: {report['status'].upper()}")
    print(f"Logged to WhyLabs: {'Yes' if report.get('whylabs_logged', False) else 'No'}")
    
    if 'performance' in report:
        print("\nPerformance Metrics:")
        for metric, value in report['performance'].items():
            print(f"{metric.capitalize()}: {value:.4f}")
    
    if report.get('alerts'):
        print("\n🚨 ALERTS:")
        for alert in report['alerts']:
            print(f"- {alert}")
    
    print("\nMonitoring complete")
