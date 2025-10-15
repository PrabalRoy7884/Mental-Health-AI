"""
Helper utilities for Mental Health in Tech Survey project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import json
import os

def load_config(config_path='config.json'):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default settings.")
        return {}

def save_config(config, config_path='config.json'):
    """Save configuration to JSON file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def create_directories(directories):
    """Create directories if they don't exist"""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directory created: {directory}")

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    return cm

def plot_roc_curve(y_true, y_proba, title="ROC Curve"):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_feature_importance(importance_df, title="Feature Importance", top_n=15):
    """Plot feature importance"""
    top_features = importance_df.head(top_n)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_features, 
                x=top_features.columns[1], y=top_features.columns[0])
    plt.title(f'{title} - Top {top_n} Features')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

def create_interactive_plot(df, x_col, y_col, color_col=None, title="Interactive Plot"):
    """Create interactive plotly plot"""
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
    fig.show()
    return fig

def create_dashboard_metrics(y_true, y_pred, y_proba=None):
    """Create comprehensive metrics for dashboard"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    return metrics

def format_prediction_result(prediction_result):
    """Format prediction result for display"""
    formatted = {
        'prediction': prediction_result['prediction'],
        'confidence': None
    }
    if 'probabilities' in prediction_result:
        probs = prediction_result['probabilities']
        max_prob = max(probs.values())
        formatted['confidence'] = f"{max_prob:.2%}"
        formatted['probabilities'] = {k: f"{v:.2%}" for k, v in probs.items()}
    return formatted

def validate_input_data(input_data, required_fields=None):
    """Validate input data for prediction"""
    if required_fields is None:
        required_fields = ['Age', 'Gender']
    missing_fields = []
    for field in required_fields:
        if field not in input_data or input_data[field] is None or input_data[field] == '':
            missing_fields.append(field)
    return len(missing_fields) == 0, missing_fields

def create_age_groups(ages):
    """Create age groups from age values"""
    return pd.cut(ages, bins=[0, 25, 35, 45, 100], 
                  labels=['18-25', '26-35', '36-45', '45+'])

def standardize_gender_input(gender):
    """Standardize gender input for consistency"""
    if pd.isna(gender) or gender == '':
        return 'Other'
    gender = str(gender).lower().strip()
    if gender in ['male', 'm', 'man']:
        return 'Male'
    elif gender in ['female', 'f', 'woman']:
        return 'Female'
    else:
        return 'Other'

def get_model_summary(results_df):
    """Get model performance summary"""
    summary = {
        'best_model': results_df.index[0],
        'best_accuracy': results_df.iloc[0]['Test Accuracy'],
        'total_models': len(results_df),
        'average_accuracy': results_df['Test Accuracy'].mean(),
        'accuracy_std': results_df['Test Accuracy'].std()
    }
    return summary

def export_results_to_csv(results_df, filename='model_results.csv'):
    """Export results to CSV file"""
    results_df.to_csv(filename)
    print(f"✅ Results exported to {filename}")

def create_prediction_explanation(prediction_result, feature_importance=None):
    """Create explanation for prediction result"""
    explanation = {
        'prediction': prediction_result['prediction'],
        'reasoning': []
    }
    if 'probabilities' in prediction_result:
        pred_class = prediction_result['prediction']
        confidence = prediction_result['probabilities'][pred_class]
        if confidence > 0.8:
            explanation['reasoning'].append("High confidence prediction")
        elif confidence > 0.6:
            explanation['reasoning'].append("Moderate confidence prediction")
        else:
            explanation['reasoning'].append("Low confidence prediction")
    if feature_importance is not None:
        top_features = feature_importance.head(3)['feature'].tolist()
        explanation['key_factors'] = top_features
    return explanation

def calculate_prediction_statistics(predictions):
    """Calculate statistics for a batch of predictions"""
    pred_values = [p['prediction'] for p in predictions]
    stats = {
        'total_predictions': len(predictions),
        'treatment_predicted': pred_values.count('Yes'),
        'no_treatment_predicted': pred_values.count('No'),
        'treatment_rate': pred_values.count('Yes') / len(predictions) * 100
    }
    return stats

def log_prediction(input_data, prediction_result, log_file='predictions.log'):
    """Log prediction for monitoring"""
    import datetime
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'input': input_data,
        'prediction': prediction_result
    }
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

class DataValidator:
    """Data validation utilities"""
    @staticmethod
    def validate_age(age):
        try:
            age = float(age)
            if 18 <= age <= 100:
                return True, age
            else:
                return False, "Age must be between 18 and 100"
        except (ValueError, TypeError):
            return False, "Age must be a number"
    
    @staticmethod
    def validate_required_fields(data, required_fields):
        missing = []
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == '':
                missing.append(field)
        return len(missing) == 0, missing
    
    @staticmethod
    def validate_categorical_field(value, allowed_values):
        if value in allowed_values:
            return True, value
        else:
            return False, f"Value must be one of: {allowed_values}"

def create_summary_report(model_results, feature_importance, test_accuracy):
    """Create a summary report of the model performance"""
    report = f"""
MENTAL HEALTH PREDICTION MODEL - SUMMARY REPORT
==============================================

MODEL PERFORMANCE:
- Best Model: {model_results.index[0]}
- Test Accuracy: {test_accuracy:.4f}
- Number of models tested: {len(model_results)}

TOP 5 IMPORTANT FEATURES:
"""
    if feature_importance is not None:
        for i, (_, row) in enumerate(feature_importance.head().iterrows(), 1):
            report += f"- {i}. {row['feature']}: {row.iloc[1]:.4f}\n"
    report += """
RECOMMENDATIONS:
- Regular model retraining is recommended
- Monitor prediction accuracy over time
- Consider collecting additional relevant features
"""
    return report
