"""
Model training utilities for Mental Health in Tech Survey
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import json
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=False),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Naive Bayes': GaussianNB()
        }
        
        self.trained_models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate performance"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                y_test_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_test_proba = None
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, average='weighted')
            recall = recall_score(y_test, y_test_pred, average='weighted')
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            # ROC AUC if probabilities available
            roc_auc = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else None
            
            # Store results
            self.results[name] = {
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc,
                'Overfitting': train_accuracy - test_accuracy
            }
            
            # Store trained model
            self.trained_models[name] = model
            
            print(f"✅ {name} completed - Test Accuracy: {test_accuracy:.4f}")
        
        # Find best model
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('Test Accuracy', ascending=False)
        
        self.best_model_name = results_df.index[0]
        self.best_model = self.trained_models[self.best_model_name]
        
        return results_df
    
    def hyperparameter_tuning(self, X_train, y_train, model_name=None):
        """Perform hyperparameter tuning for specified model"""
        if model_name is None:
            model_name = self.best_model_name
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return None
        
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.trained_models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update best model
        tuned_model = grid_search.best_estimator_
        self.trained_models[model_name + '_tuned'] = tuned_model
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return tuned_model
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from best model"""
        if self.best_model is None:
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        elif hasattr(self.best_model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': np.abs(self.best_model.coef_[0])
            }).sort_values('coefficient', ascending=False)
            return importance_df
        else:
            return None
    
    def save_models(self, path='../models/'):
        """Save all trained models"""
        os.makedirs(path, exist_ok=True)
        
        # Save best model
        joblib.dump(self.best_model, f'{path}/best_model.pkl')
        
        # Save all models
        for name, model in self.trained_models.items():
            filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            joblib.dump(model, f'{path}/{filename}_model.pkl')
        
        # Save results
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(f'{path}/model_comparison_results.csv')
        
        # Save model metadata
        model_metadata = {
            'best_model_name': self.best_model_name,
            'best_model_accuracy': float(results_df.loc[self.best_model_name, 'Test Accuracy']),
            'all_models': list(self.trained_models.keys())
        }
        
        with open(f'{path}/model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        print(f"✅ Models saved to {path}")
    
    def load_best_model(self, path='../models/'):
        """Load the best trained model"""
        self.best_model = joblib.load(f'{path}/best_model.pkl')
        
        # Load metadata
        with open(f'{path}/model_metadata.json', 'r') as f:
            metadata = json.load(f)
            self.best_model_name = metadata['best_model_name']
        
        return self.best_model
    
    def cross_validate(self, X, y, model_name=None, cv=5):
        """Perform cross-validation"""
        if model_name is None:
            model = self.best_model
            name = self.best_model_name
        else:
            model = self.trained_models[model_name]
            name = model_name
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation results for {name}:")
        print(f"CV Scores: {cv_scores}")
        print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores

def train_mental_health_model(X_train, X_test, y_train, y_test, save_path='../models/'):
    """Complete model training pipeline"""
    trainer = ModelTrainer()
    
    # Train all models
    results_df = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Perform hyperparameter tuning on best model
    tuned_model = trainer.hyperparameter_tuning(X_train, y_train)
    
    # Save models
    trainer.save_models(save_path)
    
    return trainer, results_df
