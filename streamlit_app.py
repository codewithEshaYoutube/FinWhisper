"""
FinWhisper: AI-Powered Finance Workflow Error Detection & Prediction
Complete integrated solution with dashboard, alerts, and IBM watsonx Orchestrate integration
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== DATA INGESTION ====================
class DataIngestionEngine:
    """Handles data ingestion from CSV, Excel, and API sources"""
    
    @staticmethod
    def from_csv(file_path):
        """Load data from CSV"""
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(data)} records from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return None
    
    @staticmethod
    def from_excel(file_path, sheet_name=0):
        """Load data from Excel"""
        try:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            logger.info(f"Loaded {len(data)} records from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading Excel: {e}")
            return None
    
    @staticmethod
    def generate_dummy_workflow():
        """Generate synthetic financial workflow data for testing"""
        np.random.seed(42)
        n_records = 500
        
        dates = [datetime.now() - timedelta(days=x) for x in range(n_records)]
        
        data = {
            'transaction_id': [f'TXN{i:05d}' for i in range(n_records)],
            'amount': np.random.exponential(scale=5000, size=n_records),
            'approval_time_hours': np.random.exponential(scale=2, size=n_records),
            'department': np.random.choice(['Finance', 'HR', 'Operations', 'Sales'], n_records),
            'status': np.random.choice(['Approved', 'Pending', 'Rejected'], n_records, p=[0.7, 0.2, 0.1]),
            'date': dates,
            'vendor_count': np.random.poisson(3, n_records),
            'approval_count': np.random.poisson(2, n_records),
            'notes': ['Valid transaction'] * n_records
        }
        
        # Inject anomalies
        anomaly_indices = np.random.choice(n_records, size=int(0.1 * n_records), replace=False)
        for idx in anomaly_indices:
            data['amount'][idx] = np.random.uniform(50000, 200000)
            data['approval_time_hours'][idx] = np.random.uniform(20, 50)
            data['notes'][idx] = 'Unusual pattern detected'
        
        return pd.DataFrame(data)


# ==================== ANOMALY DETECTION ====================
class AnomalyDetector:
    """Machine learning-based anomaly detection engine"""
    
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.features = None
        self.is_fitted = False
    
    def fit(self, data, feature_columns):
        """Train anomaly detection model"""
        try:
            self.features = feature_columns
            X = data[feature_columns].copy()
            X = X.fillna(X.mean())
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled)
            self.is_fitted = True
            logger.info(f"Anomaly detector trained on {len(data)} records")
        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
    
    def predict(self, data):
        """Detect anomalies in new data"""
        if not self.is_fitted:
            logger.warning("Model not fitted yet")
            return None
        
        try:
            X = data[self.features].copy()
            X = X.fillna(X.mean())
            X_scaled = self.scaler.transform(X)
            anomalies = self.model.predict(X_scaled)
            scores = self.model.score_samples(X_scaled)
            
            data['anomaly'] = (anomalies == -1).astype(int)
            data['anomaly_score'] = -scores
            
            logger.info(f"Detected {data['anomaly'].sum()} anomalies out of {len(data)} records")
            return data
        except Exception as e:
            logger.error(f"Error predicting anomalies: {e}")
            return None


# ==================== ERROR PREDICTION ====================
class ErrorPredictor:
    """Predictive model for workflow errors and bottlenecks"""
    
    @staticmethod
    def calculate_risk_score(data):
        """Calculate risk score for each transaction"""
        risk_factors = pd.DataFrame(index=data.index)
        
        # Factor 1: High amount risk
        amount_threshold = data['amount'].quantile(0.95)
        risk_factors['amount_risk'] = (data['amount'] > amount_threshold).astype(int) * 0.3
        
        # Factor 2: Approval time delay
        approval_threshold = data['approval_time_hours'].quantile(0.90)
        risk_factors['time_risk'] = (data['approval_time_hours'] > approval_threshold).astype(int) * 0.3
        
        # Factor 3: Multiple approvals needed
        risk_factors['approval_complexity'] = (data['approval_count'] > 3).astype(int) * 0.2
        
        # Factor 4: Multiple vendors
        risk_factors['vendor_risk'] = (data['vendor_count'] > 5).astype(int) * 0.2
        
        # Combined risk score
        data['risk_score'] = risk_factors.sum(axis=1)
        data['risk_level'] = pd.cut(data['risk_score'], bins=[0, 0.3, 0.6, 1.0], 
                                     labels=['Low', 'Medium', 'High'])
        
        return data
    
    @staticmethod
    def predict_bottlenecks(data):
        """Identify workflow bottlenecks"""
        bottlenecks = {
            'slow_approvals': data[data['approval_time_hours'] > data['approval_time_hours'].quantile(0.85)],
            'pending_high_amount': data[(data['status'] == 'Pending') & (data['amount'] > data['amount'].quantile(0.75))],
            'department_bottleneck': data.groupby('department')['approval_time_hours'].mean().sort_values(ascending=False)
        }
        return bottlenecks


# ==================== ALERT SYSTEM ====================
class AlertSystem:
    """Real-time alert generation and delivery"""
    
    def __init__(self, email_config=None, slack_config=None):
        self.email_config = email_config
        self.slack_config = slack_config
        self.alerts = []
    
    def generate_alert(self, alert_type, severity, message, details=None):
        """Generate alert object"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'severity': severity,
            'message': message,
            'details': details or {}
        }
        self.alerts.append(alert)
        logger.warning(f"ALERT [{severity}]: {message}")
        return alert
    
    def send_email_alert(self, recipient, subject, body):
        """Send email alert"""
        if not self.email_config:
            logger.warning("Email config not configured")
            return False
        
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = self.email_config.get('sender')
            msg['To'] = recipient
            
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['port']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {recipient}")
            return True
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def send_slack_alert(self, message):
        """Send Slack notification (requires requests library)"""
        if not self.slack_config:
            logger.warning("Slack config not configured")
            return False
        
        logger.info(f"Slack alert would be sent: {message}")
        return True


# ==================== DASHBOARD & VISUALIZATION ====================
class DashboardGenerator:
    """Generate interactive visualizations and dashboards"""
    
    @staticmethod
    def plot_anomalies(data, output_file='anomalies_plot.html'):
        """Interactive plot of anomalies"""
        fig = px.scatter(data, x='amount', y='approval_time_hours', 
                        color='anomaly', size='risk_score',
                        hover_data=['transaction_id', 'department'],
                        title='Transaction Anomalies Detection')
        fig.write_html(output_file)
        logger.info(f"Anomaly plot saved to {output_file}")
        return fig
    
    @staticmethod
    def plot_risk_distribution(data, output_file='risk_distribution.html'):
        """Risk level distribution"""
        risk_counts = data['risk_level'].value_counts()
        fig = px.bar(x=risk_counts.index, y=risk_counts.values,
                    title='Risk Level Distribution',
                    labels={'x': 'Risk Level', 'y': 'Count'})
        fig.write_html(output_file)
        return fig
    
    @staticmethod
    def plot_department_metrics(data, output_file='department_metrics.html'):
        """Department-wise performance metrics"""
        dept_data = data.groupby('department').agg({
            'approval_time_hours': 'mean',
            'amount': 'mean',
            'transaction_id': 'count'
        }).reset_index()
        
        fig = go.Figure(data=[
            go.Bar(name='Avg Approval Time (hrs)', x=dept_data['department'], 
                   y=dept_data['approval_time_hours']),
            go.Bar(name='Avg Amount ($)', x=dept_data['department'], 
                   y=dept_data['amount'])
        ])
        fig.write_html(output_file)
        return fig
    
    @staticmethod
    def generate_summary_report(data, anomaly_detector_result):
        """Generate text summary report"""
        report = f"""
=== FinWhisper Summary Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Total Transactions Analyzed: {len(data)}
Anomalies Detected: {data['anomaly'].sum()} ({data['anomaly'].sum()/len(data)*100:.2f}%)
High-Risk Transactions: {(data['risk_level'] == 'High').sum()}

Risk Distribution:
{data['risk_level'].value_counts().to_string()}

Department Performance:
{data.groupby('department')['approval_time_hours'].mean().to_string()}

Status Summary:
{data['status'].value_counts().to_string()}

Critical Alerts:
- Transactions with anomalies require immediate review
- High-risk items may face delays or approval issues
"""
        return report


# ==================== IBM WATSONX ORCHESTRATE INTEGRATION ====================
class WatsonxOrchestrationSkill:
    """Custom skill for IBM watsonx Orchestrate integration"""
    
    def __init__(self):
        self.engine = DataIngestionEngine()
        self.detector = AnomalyDetector()
        self.predictor = ErrorPredictor()
        self.alerts = AlertSystem()
    
    def execute_skill(self, input_data):
        """Main skill execution for watsonx Orchestrate"""
        try:
            # Process data
            if isinstance(input_data, str):
                data = pd.read_json(input_data)
            else:
                data = input_data
            
            # Anomaly detection
            feature_cols = ['amount', 'approval_time_hours', 'vendor_count', 'approval_count']
            self.detector.fit(data, feature_cols)
            data = self.detector.predict(data)
            
            # Risk prediction
            data = self.predictor.calculate_risk_score(data)
            
            # Generate alerts for high-risk items
            critical_data = data[data['risk_level'] == 'High']
            if len(critical_data) > 0:
                self.alerts.generate_alert(
                    'HIGH_RISK_DETECTED',
                    'CRITICAL',
                    f"{len(critical_data)} high-risk transactions detected",
                    critical_data.to_dict()
                )
            
            return {
                'status': 'success',
                'anomalies_detected': int(data['anomaly'].sum()),
                'high_risk_count': int((data['risk_level'] == 'High').sum()),
                'processed_records': len(data),
                'data': data.to_json()
            }
        except Exception as e:
            logger.error(f"Skill execution error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_skill_definition(self):
        """Return skill definition for watsonx Orchestrate"""
        return {
            'name': 'FinWhisper Error Detection',
            'description': 'Detects anomalies and predicts errors in finance workflows',
            'inputs': {
                'data': 'Financial transaction data (CSV or JSON)',
                'type': 'file or JSON'
            },
            'outputs': {
                'status': 'Execution status',
                'anomalies_detected': 'Number of anomalies',
                'high_risk_count': 'High-risk transactions',
                'processed_records': 'Total records processed'
            }
        }


# ==================== MAIN ORCHESTRATOR ====================
class FinWhisperSystem:
    """Main orchestrator for the complete FinWhisper system"""
    
    def __init__(self):
        self.data = None
        self.engine = DataIngestionEngine()
        self.detector = AnomalyDetector()
        self.predictor = ErrorPredictor()
        self.alerts = AlertSystem()
        self.dashboard = DashboardGenerator()
    
    def run_complete_analysis(self, data_source, source_type='csv'):
        """Execute complete analysis pipeline"""
        logger.info("Starting FinWhisper analysis pipeline...")
        
        # Data ingestion
        if source_type == 'csv':
            self.data = self.engine.from_csv(data_source)
        elif source_type == 'excel':
            self.data = self.engine.from_excel(data_source)
        elif source_type == 'dummy':
            self.data = self.engine.generate_dummy_workflow()
        
        if self.data is None:
            logger.error("Failed to load data")
            return None
        
        # Anomaly detection
        feature_cols = ['amount', 'approval_time_hours', 'vendor_count', 'approval_count']
        self.detector.fit(self.data, feature_cols)
        self.data = self.detector.predict(self.data)
        
        # Error prediction
        self.data = self.predictor.calculate_risk_score(self.data)
        
        # Identify bottlenecks
        bottlenecks = self.predictor.predict_bottlenecks(self.data)
        
        # Generate alerts
        if self.data['anomaly'].sum() > 0:
            self.alerts.generate_alert(
                'ANOMALIES_DETECTED',
                'HIGH',
                f"Detected {self.data['anomaly'].sum()} anomalies",
                {'count': int(self.data['anomaly'].sum())}
            )
        
        # Generate dashboards
        self.dashboard.plot_anomalies(self.data)
        self.dashboard.plot_risk_distribution(self.data)
        self.dashboard.plot_department_metrics(self.data)
        
        # Generate report
        report = self.dashboard.generate_summary_report(self.data, None)
        print(report)
        
        logger.info("Analysis pipeline completed successfully")
        return {
            'data': self.data,
            'bottlenecks': bottlenecks,
            'alerts': self.alerts.alerts,
            'report': report
        }


# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    # Initialize system
    finwhisper = FinWhisperSystem()
    
    # Run analysis on dummy data
    results = finwhisper.run_complete_analysis('dummy', source_type='dummy')
    
    # Display results
    print("\n=== ANOMALOUS TRANSACTIONS ===")
    anomalies = finwhisper.data[finwhisper.data['anomaly'] == 1]
    print(anomalies[['transaction_id', 'amount', 'approval_time_hours', 'anomaly_score', 'risk_level']])
    
    print("\n=== HIGH-RISK TRANSACTIONS ===")
    high_risk = finwhisper.data[finwhisper.data['risk_level'] == 'High']
    print(high_risk[['transaction_id', 'amount', 'risk_score', 'department']])
    
    # Initialize Orchestration Skill for watsonx
    skill = WatsonxOrchestrationSkill()
    print("\n=== WATSONX SKILL DEFINITION ===")
    print(json.dumps(skill.get_skill_definition(), indent=2))
