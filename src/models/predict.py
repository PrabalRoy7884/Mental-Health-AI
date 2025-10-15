"""
Prediction utilities for Mental Health in Tech Survey
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

# Handle imports for both direct execution and module import
try:
    from ..data.preprocess import DataPreprocessor
except ImportError:
    # If relative import fails, try absolute import
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.data.preprocess import DataPreprocessor

class MentalHealthPredictor:
    def __init__(self, model_path='../models/'):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.feature_names = []
        self.target_encoder = None
        
    def load_model_and_preprocessors(self):
        """Load trained model and all preprocessors"""
        try:
            # Load the best model
            self.model = joblib.load(f'{self.model_path}/best_model.pkl')
            
            # Load scaler
            self.scaler = joblib.load(f'{self.model_path}/feature_scaler.pkl')
            
            # Load feature names
            with open(f'{self.model_path}/selected_features.json', 'r') as f:
                self.feature_names = json.load(f)
            
            # Create simple target encoder
            self.target_encoder = type('TargetEncoder', (), {
                'inverse_transform': lambda self, x: ['No' if val == 0 else 'Yes' for val in x],
                'classes_': np.array(['No', 'Yes'])
            })()
            
            print("✅ Model and preprocessors loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("Input data must be a dictionary or pandas DataFrame")
        
        # Create feature vector with default values
        feature_vector = np.zeros(len(self.feature_names))
        
        # Feature mapping based on the training data patterns
        feature_mapping = {
            'Gender': {'Male': 0.389614, 'Female': -1.902555, 'Other': 0.0},
            'family_history': {'Yes': 1.247, 'No': -0.800912, "Don't know": 0.0},
            'work_interfere': {'Often': 1.5, 'Sometimes': 0.5, 'Rarely': -0.5, 'Never': -1.5, "Don't know": 0.0},
            'benefits': {'Yes': 1.2, 'No': -0.8, "Don't know": 0.0},
            'care_options': {'Yes': 1.1, 'No': -0.9, 'Not sure': 0.0},
            'wellness_program': {'Yes': 1.0, 'No': -1.0, "Don't know": 0.0},
            'anonymity': {'Yes': 1.0, 'No': -1.0, "Don't know": 0.0},
            'obs_consequence': {'Yes': 1.0, 'No': -1.0}
        }
        
        # Map input features to the feature vector
        for i, feature in enumerate(self.feature_names):
            if feature in df.columns and not pd.isna(df[feature].iloc[0]):
                value = df[feature].iloc[0]
                if feature in feature_mapping:
                    if isinstance(value, str) and value in feature_mapping[feature]:
                        feature_vector[i] = feature_mapping[feature][value]
                    else:
                        feature_vector[i] = float(value) if str(value).replace('.', '').replace('-', '').isdigit() else 0.0
                else:
                    # Handle encoded features (binary features)
                    if isinstance(value, (int, float)):
                        feature_vector[i] = float(value)
                    else:
                        feature_vector[i] = 0.0
        
        return feature_vector.reshape(1, -1)
    
    def predict(self, input_data, return_proba=False):
        """Make prediction on input data"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_preprocessors() first.")
        
        X_processed = self.preprocess_input(input_data)
        X_scaled = self.scaler.transform(X_processed)
        
        prediction = self.model.predict(X_scaled)
        prediction_labels = self.target_encoder.inverse_transform(prediction)
        
        result = {
            'prediction': prediction_labels[0] if len(prediction_labels) == 1 else prediction_labels,
            'prediction_encoded': int(prediction[0]) if len(prediction) == 1 else prediction.tolist()
        }
        
        if return_proba and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
            prob_dict = {}
            for i, class_label in enumerate(self.target_encoder.classes_):
                prob_dict[class_label] = float(probabilities[0][i]) if len(probabilities) == 1 else probabilities[:, i].tolist()
            result['probabilities'] = prob_dict
        
        return result
    
    def predict_batch(self, input_data_list, return_proba=False):
        """Make predictions on a batch of input data"""
        results = []
        for input_data in input_data_list:
            result = self.predict(input_data, return_proba)
            results.append(result)
        return results
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return "No model loaded"
        
        info = {
            'model_type': type(self.model).__name__,
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'target_classes': self.target_encoder.classes_.tolist() if self.target_encoder else None
        }
        
        return info
    
    def get_feature_importance(self, top_n=10):
        """Get feature importance from the model"""
        if self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df.head(top_n)
            
        elif hasattr(self.model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': np.abs(self.model.coef_[0])
            }).sort_values('coefficient', ascending=False)
            return importance_df.head(top_n)
        
        else:
            return None

def create_sample_input():
    """Create a sample input for testing"""
    sample_input = {
        'Age': 28,
        'Gender': 'Male',
        'Country': 'United States',
        'state': 'CA',
        'self_employed': 'No',
        'family_history': 'No',
        'treatment': 'Yes',  # Ignored during prediction
        'work_interfere': 'Sometimes',
        'no_employees': '6-25',
        'remote_work': 'No',
        'tech_company': 'Yes',
        'benefits': 'Yes',
        'care_options': 'Not sure',
        'wellness_program': 'No',
        'seek_help': 'Yes',
        'anonymity': 'Yes',
        'leave': 'Somewhat easy',
        'mental_health_consequence': 'No',
        'phys_health_consequence': 'No',
        'coworkers': 'Some of them',
        'supervisor': 'Yes',
        'mental_health_interview': 'No',
        'phys_health_interview': 'Maybe',
        'mental_vs_physical': "Don't know",
        'obs_consequence': 'No'
    }
    return sample_input

def test_prediction(model_path='../models/'):
    """Test the prediction functionality"""
    predictor = MentalHealthPredictor(model_path)
    
    if not predictor.load_model_and_preprocessors():
        return None
    
    sample_input = create_sample_input()
    result = predictor.predict(sample_input, return_proba=True)
    
    print("Sample prediction result:")
    print(f"Input: {sample_input}")
    print(f"Prediction: {result}")
    
    return predictor
