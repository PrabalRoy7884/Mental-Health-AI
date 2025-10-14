import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Advanced models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Page configuration
st.set_page_config(
    page_title="Mental Health Treatment Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        color: #000000;
    }
    .metric-container {
        color: #000000 !important;
    }
    .metric-value {
        color: #000000 !important;
    }
    .metric-label {
        color: #000000 !important;
    }
    div[data-testid="metric-container"] {
        color: #000000 !important;
    }
    div[data-testid="metric-container"] .metric-value {
        color: #000000 !important;
    }
    div[data-testid="metric-container"] .metric-label {
        color: #000000 !important;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 1rem;
        border: 2px solid #1f77b4;
        text-align: center;
        margin: 2rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-color: #f44336;
        color: #c62828;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-color: #4caf50;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the processed data"""
    try:
        train_data = pd.read_csv('data/processed/train_data.csv')
        test_data = pd.read_csv('data/processed/test_data.csv')
        complete_data = pd.read_csv('data/processed/complete_processed_data.csv')
        return train_data, test_data, complete_data
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}. Please ensure the data is processed.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data
def load_preprocessing_models():
    """Load preprocessing models"""
    try:
        scaler = joblib.load('models/feature_scaler.pkl')
        selector = joblib.load('models/feature_selector.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        target_encoder = joblib.load('models/target_encoder.pkl')
        
        with open('data/processed/selected_features.json', 'r') as f:
            selected_features = json.load(f)
        
        return scaler, selector, label_encoders, target_encoder, selected_features
    except FileNotFoundError as e:
        st.error(f"Preprocessing models not found: {e}")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading preprocessing models: {e}")
        return None, None, None, None, None

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and return the best one"""
    
    models = {
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False),
        'XGBoost': XGBClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f'Training {name}...')
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        trained_models[name] = model
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text('Training completed!')
    progress_bar.empty()
    status_text.empty()
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = trained_models[best_model_name]
    
    return trained_models, results, best_model_name, best_model

def create_user_input_form():
    """Create user input form for mental health survey"""
    st.markdown('<div class="sub-header">📋 Mental Health Survey</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Personal Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Your current age")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], help="Your gender identity")
        country = st.selectbox("Country", [
            "United States", "Canada", "United Kingdom", "Germany", "Netherlands",
            "Australia", "France", "India", "Other"
        ], help="Your country of residence")
        self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"], help="Are you currently self-employed?")
        family_history = st.selectbox("Do you have a family history of mental illness?", 
                                    ["Yes", "No", "Don't know"], 
                                    help="Do you have a family history of mental health issues?")
        work_interfere = st.selectbox("If you have a mental health condition, do you feel that it interferes with your work?", 
                                    ["Often", "Rarely", "Never", "Sometimes", "Don't know"], 
                                    help="How often does mental health interfere with your work?")
    
    with col2:
        st.markdown("#### Work Information")
        tech_company = st.selectbox("Do you work for a tech company?", ["Yes", "No"], help="Do you work in the technology industry?")
        no_employees = st.selectbox("How many employees does your company have?", [
            "1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"
        ], help="What is the size of your company?")
        remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"], help="Do you work remotely (from home or other locations)?")
        benefits = st.selectbox("Does your employer provide mental health benefits?", 
                              ["Yes", "No", "Don't know"], 
                              help="Does your employer provide mental health benefits?")
        wellness_program = st.selectbox("Has your employer ever discussed mental health as part of an employee wellness program?", 
                                      ["Yes", "No", "Don't know"], 
                                      help="Has your employer discussed mental health in wellness programs?")
        seek_help = st.selectbox("Does your employer provide resources to learn more about mental health issues and how to seek help?", 
                               ["Yes", "No", "Don't know"], 
                               help="Does your employer provide mental health resources and education?")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### Work Environment")
        anonymity = st.selectbox("Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?", 
                               ["Yes", "No", "Don't know"], 
                               help="Is your privacy protected when seeking mental health treatment?")
        leave = st.selectbox("How easy is it for you to take medical leave for a mental health condition?", 
                           ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"], 
                           help="How easy is it to take leave for mental health reasons?")
        care_options = st.selectbox("Do you know the options for mental health care your employer provides?", 
                                  ["Yes", "No", "Not sure"], 
                                  help="Are you aware of mental health care options provided by your employer?")
        obs_consequence = st.selectbox("Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?", 
                                     ["Yes", "No"], 
                                     help="Have you seen negative consequences for coworkers with mental health conditions?")
    
    with col4:
        st.markdown("#### Attitudes & Perceptions")
        mental_health_consequence = st.selectbox("Do you think that discussing a mental health issue with your employer would have negative consequences?", 
                                               ["Yes", "No", "Maybe"], 
                                               help="Do you think discussing mental health with your employer would have negative consequences?")
        coworkers = st.selectbox("Would you be willing to discuss a mental health issue with your coworkers?", 
                               ["Yes", "No", "Some of them"], 
                               help="Would you discuss mental health issues with coworkers?")
        supervisor = st.selectbox("Would you be willing to discuss a mental health issue with your direct supervisor(s)?", 
                                ["Yes", "No", "Some of them"], 
                                help="Would you discuss mental health issues with your supervisor?")
        mental_health_interview = st.selectbox("Would you bring up a mental health issue with a potential employer in an interview?", 
                                             ["Yes", "No", "Maybe"], 
                                             help="Would you discuss mental health in a job interview?")
        mental_vs_physical = st.selectbox("Do you feel that your employer takes mental health as seriously as physical health?", 
                                        ["Yes", "No", "Don't know"], 
                                        help="Does your employer treat mental health as seriously as physical health?")
    
    return {
        'Age': age,
        'Gender': gender,
        'Country': country,
        'self_employed': self_employed,
        'family_history': family_history,
        'work_interfere': work_interfere,
        'no_employees': no_employees,
        'remote_work': remote_work,
        'tech_company': tech_company,
        'benefits': benefits,
        'wellness_program': wellness_program,
        'seek_help': seek_help,
        'anonymity': anonymity,
        'leave': leave,
        'care_options': care_options,
        'obs_consequence': obs_consequence,
        'mental_health_consequence': mental_health_consequence,
        'coworkers': coworkers,
        'supervisor': supervisor,
        'mental_health_interview': mental_health_interview,
        'mental_vs_physical': mental_vs_physical
    }

def preprocess_user_input(user_input, label_encoders, scaler, selector, selected_features):
    """Preprocess user input for prediction"""
    try:
        # Create DataFrame from user input
        df = pd.DataFrame([user_input])
        
        # Encode using provided label encoders when available
        for col in df.columns:
            if col in label_encoders:
                try:
                    if df[col].iloc[0] not in label_encoders[col].classes_:
                        st.error(f"Error encoding {col}: Value '{df[col].iloc[0]}' not found in training data")
                        st.info(f"Available values for {col}: {list(label_encoders[col].classes_)}")
                        return None
                    df[col] = label_encoders[col].transform(df[col])
                except ValueError as e:
                    st.error(f"Error encoding {col}: {e}")
                    st.info(f"Available values for {col}: {list(label_encoders[col].classes_)}")
                    return None
        
        # Create specific dummy/derived variables used by the model
        if 'family_history' in df.columns:
            df['family_history_encoded'] = 1 if df['family_history'].iloc[0] == 'Yes' else 0
        if 'obs_consequence' in df.columns:
            df['obs_consequence_encoded'] = 1 if df['obs_consequence'].iloc[0] == 'Yes' else 0
        if 'Gender' in df.columns:
            df['Gender_Male'] = 1 if df['Gender'].iloc[0] == 'Male' else 0
        if 'work_interfere' in df.columns:
            df['work_interfere_Often'] = 1 if df['work_interfere'].iloc[0] == 'Often' else 0
            df['work_interfere_Rarely'] = 1 if df['work_interfere'].iloc[0] == 'Rarely' else 0
        if 'benefits' in df.columns:
            df['benefits_Yes'] = 1 if df['benefits'].iloc[0] == 'Yes' else 0
        if 'care_options' in df.columns:
            df['care_options_Not sure'] = 1 if df['care_options'].iloc[0] == 'Not sure' else 0
            df['care_options_Yes'] = 1 if df['care_options'].iloc[0] == 'Yes' else 0
        if 'anonymity' in df.columns:
            df['anonymity_Yes'] = 1 if df['anonymity'].iloc[0] == 'Yes' else 0
        if 'leave' in df.columns:
            df['leave_Somewhat difficult'] = 1 if df['leave'].iloc[0] == 'Somewhat difficult' else 0
        if 'mental_health_consequence' in df.columns:
            df['mental_health_consequence_No'] = 1 if df['mental_health_consequence'].iloc[0] == 'No' else 0
            df['mental_health_consequence_Yes'] = 1 if df['mental_health_consequence'].iloc[0] == 'Yes' else 0
        
        # Align to model's expected selected features
        model_features_df = pd.DataFrame(0, index=[0], columns=selected_features)
        for feature in selected_features:
            if feature in df.columns:
                model_features_df[feature] = df[feature].values[0]
        
        # Scale features
        df_scaled = scaler.transform(model_features_df)
        return df_scaled
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        st.error(f"Expected features: {selected_features}")
        st.error(f"Available features: {list(df.columns) if 'df' in locals() else 'None'}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">🧠 Mental Health Treatment Prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
        Predict the likelihood of seeking mental health treatment based on workplace and personal factors
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    train_data, test_data, complete_data = load_data()
    if train_data is None:
        st.stop()
    
    # Load preprocessing models
    scaler, selector, label_encoders, target_encoder, selected_features = load_preprocessing_models()
    if scaler is None:
        st.stop()
    
    # Sidebar navigation with unique key
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        [
            "🏠 Home & Prediction",
            "📊 Data Analysis",
            "🤖 Model Performance",
            "📈 Insights & Recommendations",
        ],
        key="nav_page_app2",
    )
    
    if page == "🏠 Home & Prediction":
        st.markdown('<div class="sub-header">🔮 Mental Health Treatment Prediction</div>', unsafe_allow_html=True)
        
        # User input form
        user_input = create_user_input_form()
        
        if st.button("🔮 Predict Treatment Likelihood", type="primary", use_container_width=True):
            try:
                # Preprocess input
                processed_input = preprocess_user_input(user_input, label_encoders, scaler, selector, selected_features)
                
                if processed_input is None:
                    st.error("Failed to preprocess input. Please check your data.")
                    return
                
                # Prepare training data
                X_train = train_data.drop('target', axis=1)
                y_train = train_data['target']
                X_test = test_data.drop('target', axis=1)
                y_test = test_data['target']
                
                # Train models and get predictions
                with st.spinner("Training models and making prediction..."):
                    trained_models, results, best_model_name, best_model = train_models(X_train, y_train, X_test, y_test)
                    
                    # Make prediction
                    prediction_proba = best_model.predict_proba(processed_input)[0]
                    prediction = best_model.predict(processed_input)[0]
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                return
            
            # Display results
            st.markdown('<div class="sub-header">🎯 Detailed Prediction Results</div>', unsafe_allow_html=True)
            
            # Main prediction metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Model", best_model_name)
                st.metric("Model Accuracy", f"{results[best_model_name]['accuracy']:.1%}")
            
            with col2:
                treatment_prob = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
                st.metric("Treatment Probability", f"{treatment_prob:.1%}")
            
            with col3:
                no_treatment_prob = prediction_proba[0] if len(prediction_proba) > 1 else 1 - prediction_proba[0]
                st.metric("No Treatment Probability", f"{no_treatment_prob:.1%}")
            
            # Removed Risk Level metric display per request
            
            # Detailed prediction display
            st.markdown("---")
            
            # Main prediction result
            prediction_class = "high-risk" if treatment_prob > 0.6 else "low-risk"
            prediction_text = "Likely to seek treatment" if prediction == 1 else "Unlikely to seek treatment"
            confidence_level = "High" if max(treatment_prob, no_treatment_prob) > 0.8 else "Medium" if max(treatment_prob, no_treatment_prob) > 0.6 else "Low"
            
            st.markdown(f"""
            <div class="prediction-box {prediction_class}">
                <h2>🎯 {prediction_text}</h2>
                <p><strong>Confidence Level:</strong> {confidence_level} ({max(treatment_prob, no_treatment_prob):.1%})</p>
                <p><strong>Model Used:</strong> {best_model_name}</p>
                <p><strong>Model Performance:</strong> {results[best_model_name]['accuracy']:.1%} accuracy</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability visualization
            st.markdown('<div class="sub-header">📊 Probability Breakdown</div>', unsafe_allow_html=True)
            
            prob_data = {
                'Outcome': ['Will Seek Treatment', 'Will Not Seek Treatment'],
                'Probability': [treatment_prob, no_treatment_prob],
                'Color': ['#1f77b4', '#ff7f0e']
            }
            
            fig_prob = go.Figure(data=[
                go.Bar(name='Treatment Likelihood', 
                      x=prob_data['Outcome'], 
                      y=prob_data['Probability'],
                      marker_color=prob_data['Color'],
                      text=[f"{p:.1%}" for p in prob_data['Probability']],
                      textposition='auto')
            ])
            fig_prob.update_layout(
                title="Treatment Seeking Probability Distribution",
                yaxis_title="Probability",
                yaxis=dict(tickformat='.0%', range=[0, 1]),
                height=400
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # Model performance metrics
            st.markdown('<div class="sub-header">📈 Model Performance Details</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Precision", f"{results[best_model_name]['precision']:.1%}")
            with col2:
                st.metric("Recall", f"{results[best_model_name]['recall']:.1%}")
            with col3:
                st.metric("F1 Score", f"{results[best_model_name]['f1']:.1%}")
            with col4:
                st.metric("ROC AUC", f"{results[best_model_name]['roc_auc']:.1%}")
            
            # Feature importance (if available)
            if hasattr(best_model, 'feature_importances_'):
                st.markdown('<div class="sub-header">🔍 Feature Importance Analysis</div>', unsafe_allow_html=True)
                feature_importance = pd.DataFrame({
                    'feature': selected_features,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Top features
                top_features = feature_importance.head(10)
                
                fig = px.bar(top_features, x='importance', y='feature', 
                           orientation='h', title="Top 10 Most Important Features in This Prediction")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance table
                st.markdown("#### Detailed Feature Importance")
                st.dataframe(feature_importance, use_container_width=True)
            
            # Detailed recommendations based on prediction
            st.markdown('<div class="sub-header">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
            
            if prediction == 1:  # Likely to seek treatment
                st.success("""
                **🎯 Based on your profile, you are likely to seek mental health treatment.**
                
                **This is positive because:**
                - You recognize the importance of mental health care
                - You have supportive workplace factors
                - You're proactive about your mental well-being
                
                **Recommendations for you:**
                - ✅ Continue prioritizing your mental health
                - ✅ Utilize available workplace mental health resources
                - ✅ Maintain open communication with support systems
                - ✅ Consider preventive mental health practices
                - ✅ Help reduce stigma by being open about mental health
                """)
            else:  # Unlikely to seek treatment
                st.warning("""
                **⚠️ Based on your profile, you may be less likely to seek mental health treatment.**
                
                **This could be due to:**
                - Limited access to mental health resources
                - Workplace stigma or lack of support
                - Personal barriers to seeking help
                
                **Recommendations for you:**
                - 🔍 Explore available mental health resources
                - 💬 Consider talking to trusted colleagues or friends
                - 🏥 Research local mental health services
                - 📚 Educate yourself about mental health
                - 🤝 Seek support from employee assistance programs
                """)
            
            # Additional insights
            st.markdown('<div class="sub-header">🔍 Additional Insights</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📊 Your Profile Analysis:**
                - **Age**: {age} years old
                - **Gender**: {gender}
                - **Country**: {country}
                - **Employment**: {employment_status}
                - **Company Size**: {company_size}
                """.format(
                    age=user_input['Age'],
                    gender=user_input['Gender'],
                    country=user_input['Country'],
                    employment_status="Self-employed" if user_input['self_employed'] == 'Yes' else "Employed",
                    company_size=user_input['no_employees']
                ))
            
            with col2:
                st.markdown("""
                **🏢 Workplace Factors:**
                - **Mental Health Benefits**: {benefits}
                - **Wellness Program**: {wellness}
                - **Remote Work**: {remote}
                - **Tech Company**: {tech}
                - **Work Interference**: {interference}
                """.format(
                    benefits=user_input['benefits'],
                    wellness=user_input['wellness_program'],
                    remote=user_input['remote_work'],
                    tech=user_input['tech_company'],
                    interference=user_input['work_interfere']
                ))
            
            # Disclaimer
            st.markdown("""
            ---
            **⚠️ Important Disclaimer:**
            This prediction is based on statistical analysis and should not replace professional medical advice. 
            If you're experiencing mental health concerns, please consult with a qualified healthcare provider.
            """)

    elif page == "📊 Data Analysis":
        st.markdown('<div class="sub-header">📊 Dataset Analysis</div>', unsafe_allow_html=True)
        try:
            original_data = pd.read_csv('data/raw/mental_health_survey.csv')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Responses", len(original_data))
            with col2:
                treatment_rate = original_data['treatment'].value_counts(normalize=True).get('Yes', 0) * 100
                st.metric("Treatment Rate", f"{treatment_rate:.1f}%")
            with col3:
                avg_age = original_data['Age'].mean()
                st.metric("Average Age", f"{avg_age:.1f}")
            with col4:
                tech_company_rate = original_data['tech_company'].value_counts(normalize=True).get('Yes', 0) * 100
                st.metric("Tech Company Rate", f"{tech_company_rate:.1f}%")

            st.markdown('<div class="sub-header">📈 Treatment Distribution</div>', unsafe_allow_html=True)
            treatment_counts = original_data['treatment'].value_counts()
            c1, c2 = st.columns(2)
            with c1:
                fig = px.pie(values=treatment_counts.values, names=treatment_counts.index, title="Treatment Seeking Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(x=treatment_counts.index, y=treatment_counts.values, title="Treatment Seeking Count")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="sub-header">👥 Age Distribution by Treatment Status</div>', unsafe_allow_html=True)
            fig = px.box(original_data, x='treatment', y='Age', title="Age Distribution by Treatment Seeking Behavior")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="sub-header">⚥ Gender Analysis</div>', unsafe_allow_html=True)
            gender_treatment = pd.crosstab(original_data['Gender'], original_data['treatment'])
            gender_treatment_pct = pd.crosstab(original_data['Gender'], original_data['treatment'], normalize='index') * 100
            g1, g2 = st.columns(2)
            with g1:
                fig = px.bar(gender_treatment, title="Treatment by Gender (Count)")
                st.plotly_chart(fig, use_container_width=True)
            with g2:
                fig = px.bar(gender_treatment_pct, title="Treatment Rate by Gender (%)")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="sub-header">💼 Work Interference Analysis</div>', unsafe_allow_html=True)
            work_treatment = pd.crosstab(original_data['work_interfere'], original_data['treatment'], normalize='index') * 100
            fig = px.bar(work_treatment, title="Treatment Rate by Work Interference Level (%)")
            st.plotly_chart(fig, use_container_width=True)
        except FileNotFoundError:
            st.error("Original data file not found.")

    elif page == "🤖 Model Performance":
        st.markdown('<div class="sub-header">🤖 Model Performance Comparison</div>', unsafe_allow_html=True)
        
        # Try to load improved results first, fallback to original
        try:
            # Load improved results
            improved_results = pd.read_csv('models/model_comparison_results_optimized.csv', index_col=0)
            st.success("✅ Showing optimized model results!")
            results_df = improved_results.round(4)
            
            # Display improvement summary
            st.info("🚀 **Model Improvements Applied:**\n- Advanced hyperparameter optimization\n- Better feature engineering\n- Ensemble methods\n- Reduced overfitting")
            
        except FileNotFoundError:
            st.warning("⚠️ Improved results not found. Training models with current settings...")
            X_train = train_data.drop('target', axis=1)
            y_train = train_data['target']
            X_test = test_data.drop('target', axis=1)
            y_test = test_data['target']
            
            if st.button("🔄 Train and Compare Models", type="primary"):
                with st.spinner("Training all models..."):
                    trained_models, results, best_model_name, best_model = train_models(X_train, y_train, X_test, y_test)
                    results_df = pd.DataFrame(results).T.round(4)
        
        # Display results
        st.markdown('<div class="sub-header">📊 Model Performance Metrics</div>', unsafe_allow_html=True)
        st.dataframe(results_df, use_container_width=True)
        
        # Performance visualizations
        pv1, pv2 = st.columns(2)
        with pv1:
            fig = px.bar(x=results_df.index, y=results_df['Test Accuracy'], 
                        title="Model Accuracy Comparison", 
                        color=results_df['Test Accuracy'],
                        color_continuous_scale='viridis')
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with pv2:
            # Handle ROC AUC (some models might not have it)
            roc_data = results_df['ROC AUC'].dropna()
            if not roc_data.empty:
                fig = px.bar(x=roc_data.index, y=roc_data.values, 
                            title="ROC AUC Comparison",
                            color=roc_data.values,
                            color_continuous_scale='plasma')
                fig.update_xaxes(tickangle=45)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ROC AUC not available for all models")
        
        # Best model summary
        best_model_name = results_df.index[0]
        best_accuracy = results_df.loc[best_model_name, 'Test Accuracy']
        best_roc_auc = results_df.loc[best_model_name, 'ROC AUC'] if pd.notna(results_df.loc[best_model_name, 'ROC AUC']) else 'N/A'
        
        st.markdown(f'<div class="sub-header">🏆 Best Model: {best_model_name}</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", f"{best_accuracy:.1%}")
        with col2:
            st.metric("Precision", f"{results_df.loc[best_model_name, 'Precision']:.1%}")
        with col3:
            st.metric("Recall", f"{results_df.loc[best_model_name, 'Recall']:.1%}")
        with col4:
            if pd.notna(best_roc_auc):
                st.metric("ROC AUC", f"{best_roc_auc:.1%}")
            else:
                st.metric("ROC AUC", "N/A")
        
        # Overfitting analysis
        st.markdown('<div class="sub-header">📈 Overfitting Analysis</div>', unsafe_allow_html=True)
        overfitting_data = results_df['Overfitting'].dropna()
        if not overfitting_data.empty:
            fig = px.bar(x=overfitting_data.index, y=overfitting_data.values,
                        title="Overfitting (Train Accuracy - Test Accuracy)",
                        color=overfitting_data.values,
                        color_continuous_scale='RdYlGn_r')
            fig.add_hline(y=0, line_dash="dash", line_color="red", 
                         annotation_text="Ideal: No Overfitting")
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance comparison table
        st.markdown('<div class="sub-header">📋 Detailed Performance Comparison</div>', unsafe_allow_html=True)
        
        # Create a styled comparison table
        comparison_data = results_df[['Test Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']].copy()
        
        # Highlight best values
        def highlight_max(s):
            return ['background-color: #d4edda' if v == s.max() else '' for v in s]
        
        st.dataframe(comparison_data.style.apply(highlight_max), use_container_width=True)

    elif page == "📈 Insights & Recommendations":
        st.markdown('<div class="sub-header">💡 Key Insights & Recommendations</div>', unsafe_allow_html=True)
        st.markdown("""
        ### 🔍 Key Insights from the Analysis:
        1. Treatment Seeking Rate: ~50% of respondents sought treatment
        2. Work Interference strongly correlates with treatment seeking
        3. Company support (benefits, care options) increases help-seeking
        4. Gender and age patterns exist across cohorts

        ### 💼 Workplace Recommendations:
        - Improve mental health benefits and awareness
        - Reduce stigma via training and leadership messaging
        - Offer flexible work arrangements and confidential support

        ### 🏥 Individual Recommendations:
        - Seek help early; leverage available benefits and EAP
        - Build support networks; practice regular self-care
        - Normalize conversations around mental health
        """)

if __name__ == "__main__":
    main()


