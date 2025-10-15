"""
Data preprocessing utilities for Mental Health in Tech Survey
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import json
import os

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_classif, k=20)
        self.target_encoder = LabelEncoder()
        self.selected_features = []
    
    def clean_age(self, df):
        """Clean and standardize age column"""
        if 'Age' in df.columns:
            # Remove invalid ages
            df['Age'] = df['Age'].apply(lambda x: np.nan if x < 18 or x > 100 else x)
            # Fill missing ages with median
            df['Age'].fillna(df['Age'].median(), inplace=True)
        return df
    
    def standardize_gender(self, gender):
        """Standardize gender responses"""
        if pd.isna(gender):
            return 'Other'
        gender = str(gender).lower().strip()
        
        # Male variations
        if gender in ['male', 'm', 'man', 'male-ish', 'maile', 'cis male', 'male (cis)', 'cis man']:
            return 'Male'
        # Female variations
        elif gender in ['female', 'f', 'woman', 'female-ish', 'femail', 'cis female', 'female (cis)', 'cis woman']:
            return 'Female'
        # Non-binary/Other variations
        else:
            return 'Other'
    
    def clean_gender(self, df):
        """Clean and standardize gender column"""
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].apply(self.standardize_gender)
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Fill categorical missing values
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'treatment':  # Don't fill target variable
                if df[col].isnull().sum() > 0:
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df[col].fillna(mode_value[0], inplace=True)
                    else:
                        df[col].fillna('Unknown', inplace=True)
        
        # Fill numerical missing values
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def encode_features(self, df, fit=True):
        """Encode categorical features"""
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable from encoding
        if 'treatment' in categorical_columns:
            categorical_columns.remove('treatment')
        
        encoded_df = df.copy()
        
        for col in categorical_columns:
            unique_values = df[col].nunique()
            
            if unique_values == 2:
                # Binary encoding
                if fit:
                    le = LabelEncoder()
                    encoded_df[col + '_encoded'] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        encoded_df[col + '_encoded'] = self.label_encoders[col].transform(df[col])
            
            elif unique_values > 2 and unique_values <= 10:
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                encoded_df = pd.concat([encoded_df, dummies], axis=1)
        
        return encoded_df
    
    def scale_features(self, X, fit=True):
        """Scale features using StandardScaler"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def select_features(self, X, y, fit=True):
        """Select top K features"""
        if fit:
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
        else:
            X_selected = self.feature_selector.transform(X)
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def preprocess_data(self, df, target_col='treatment', fit=True):
        """Complete preprocessing pipeline"""
        # Clean data
        df_clean = self.clean_age(df.copy())
        df_clean = self.clean_gender(df_clean)
        df_clean = self.handle_missing_values(df_clean)
        
        # Encode features
        df_encoded = self.encode_features(df_clean, fit=fit)
        
        # Separate features and target
        feature_columns = [col for col in df_encoded.columns if col not in [target_col]]
        X = df_encoded[feature_columns]
        
        # Handle any remaining non-numeric columns
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in non_numeric_cols:
            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Scale features
        X_scaled = self.scale_features(X, fit=fit)
        
        # Prepare target
        y = None
        if target_col in df_encoded.columns:
            if fit:
                y = self.target_encoder.fit_transform(df_encoded[target_col])
            else:
                y = self.target_encoder.transform(df_encoded[target_col])
        
        # Feature selection (only if we have target)
        if y is not None and fit:
            X_selected = self.select_features(X_scaled, y, fit=True)
        elif y is not None:
            X_selected = self.select_features(X_scaled, y, fit=False)
        else:
            X_selected = X_scaled
        
        return X_selected, y
    
    def save_preprocessors(self, path='../models/'):
        """Save all preprocessors"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.label_encoders, f'{path}/label_encoders.pkl')
        joblib.dump(self.scaler, f'{path}/feature_scaler.pkl')
        joblib.dump(self.feature_selector, f'{path}/feature_selector.pkl')
        joblib.dump(self.target_encoder, f'{path}/target_encoder.pkl')
        
        with open(f'{path}/selected_features.json', 'w') as f:
            json.dump(self.selected_features, f)
    
    def load_preprocessors(self, path='../models/'):
        """Load all preprocessors"""
        self.label_encoders = joblib.load(f'{path}/label_encoders.pkl')
        self.scaler = joblib.load(f'{path}/feature_scaler.pkl')
        self.feature_selector = joblib.load(f'{path}/feature_selector.pkl')
        self.target_encoder = joblib.load(f'{path}/target_encoder.pkl')
        
        with open(f'{path}/selected_features.json', 'r') as f:
            self.selected_features = json.load(f)

def load_and_preprocess_data(file_path, preprocessor=None, fit=True):
    """Load and preprocess data"""
    df = pd.read_csv(file_path)
    
    if preprocessor is None:
        preprocessor = DataPreprocessor()
    
    X, y = preprocessor.preprocess_data(df, fit=fit)
    
    return X, y, preprocessor
