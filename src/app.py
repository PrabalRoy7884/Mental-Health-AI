import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
from datetime import datetime
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
    page_title="Mental Health AI Predictor 🧠",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-modern CSS with animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main header with gradient */
    .main-header {
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        margin: 2rem 0;
        background: linear-gradient(45deg, #0d47a1, #1565c0, #1976d2, #0d47a1);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 8s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .subtitle {
        text-align: center;
        font-size: 1.4rem;
        color: #1a237e;
        margin-bottom: 3rem;
        font-weight: 700;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .float-element {
        animation: float 3s ease-in-out infinite;
    }

    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(13, 71, 161, 0.5); }
        50% { box-shadow: 0 0 40px rgba(13, 71, 161, 0.8); }
    }

    .glow-card {
        animation: glow 2s ease-in-out infinite;
    }

    .sub-header {
        font-size: 2rem;
        color: #0d47a1;
        margin: 2.5rem 0 1.5rem 0;
        font-weight: 800;
        padding-left: 20px;
        border-left: 6px solid #1565c0;
    }

    /* Metric cards - White text on deep blue */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #000000;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(13, 71, 161, 0.4);
        margin: 1rem 0;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
        border: 2px solid #0d47a1;
    }

    .metric-card * {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* Streamlit metric components */
    div[data-testid="metric-container"] {
        color: #000000 !important;
    }
    div[data-testid="metric-container"] .metric-value {
        color: #000000 !important;
    }
    div[data-testid="metric-container"] .metric-label {
        color: #000000 !important;
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

    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }

    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 60px rgba(13, 71, 161, 0.6);
    }

    /* Prediction box - Deep colored text */
    .prediction-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #050400 100%);
        padding: 3rem;
        border-radius: 25px;
        border: 4px solid #0d47a1;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 50px rgba(13, 71, 161, 0.3);
        position: relative;
    }

    .prediction-box * {
        color: #0d47a1 !important;
        font-weight: 800 !important;
    }

    .high-risk {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-color: #b71c1c;
    }

    .high-risk * {
        color: #b71c1c !important;
        font-weight: 800 !important;
    }

    .low-risk {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-color: #1b5e20;
    }

    .low-risk * {
        color: #1b5e20 !important;
        font-weight: 800 !important;
    }

    .prediction-box h2 {
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: 1.5rem;
    }

    /* Info cards - Deep text */
    .info-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
        margin: 1.5rem 0;
        border-top: 5px solid #0d47a1;
        transition: all 0.3s ease;
    }

    .info-card h2, .info-card h3, .info-card h4 {
        color: #0d47a1 !important;
        font-weight: 800 !important;
    }

    .info-card p {
        color: #1a237e !important;
        font-weight: 600 !important;
        line-height: 1.8;
        font-size: 1.05rem;
    }

    .info-card li {
        color: #1a237e !important;
        font-weight: 600 !important;
        line-height: 1.8;
        margin: 0.5rem 0;
    }

    .info-card strong {
        color: #01579b !important;
        font-weight: 800 !important;
    }

    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.25);
    }

    /* Alert boxes - Deep colored text */
    .alert {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.15);
        border-left: 6px solid;
    }

    .alert h2, .alert h3, .alert h4 {
        font-weight: 900 !important;
        margin-bottom: 1rem;
    }

    .alert p {
        font-weight: 600 !important;
        line-height: 1.8;
        font-size: 1.05rem;
    }

    .alert li {
        font-weight: 600 !important;
        line-height: 1.8;
        margin: 0.5rem 0;
    }

    .alert strong {
        font-weight: 900 !important;
    }

    .alert-success {
        background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%);
        border-left-color: #1b5e20;
    }

    .alert-success * {
        color: #1b5e20 !important;
    }

    .alert-warning {
        background: linear-gradient(135deg, #ffe0b2 0%, #ffcc80 100%);
        border-left-color: #e65100;
    }

    .alert-warning * {
        color: #e65100 !important;
    }

    .alert-danger {
        background: linear-gradient(135deg, #ffcdd2 0%, #ef9a9a 100%);
        border-left-color: #b71c1c;
    }

    .alert-danger * {
        color: #b71c1c !important;
    }

    .alert-info {
        background: linear-gradient(135deg, #bbdefb 0%, #90caf9 100%);
        border-left-color: #0d47a1;
    }

    .alert-info * {
        color: #0d47a1 !important;
    }

    /* Buttons - Deep blue */
    .stButton>button {
        background: linear-gradient(135deg, #0d47a1 0%, #01579b 100%);
        color: white !important;
        font-size: 1.3rem;
        font-weight: 800 !important;
        padding: 1.2rem 3rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 8px 25px rgba(13, 71, 161, 0.5);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 12px 35px rgba(13, 71, 161, 0.7);
    }

    /* Stat numbers */
    .stat-number {
        font-size: 3.5rem;
        font-weight: 900;
        color: #0d47a1 !important;
    }

    .stat-label {
        font-size: 1.1rem;
        color: #1a237e !important;
        font-weight: 700 !important;
        margin-top: 0.5rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 15px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 800 !important;
        color: #0d47a1 !important;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0d47a1 0%, #01579b 100%);
        color: white !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d47a1 0%, #01579b 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
        font-weight: 700 !important;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.6rem 1.2rem;
        border-radius: 20px;
        font-weight: 800 !important;
        font-size: 0.95rem;
        margin: 0.3rem;
    }

    .badge-success {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        color: white !important;
    }

    .badge-warning {
        background: linear-gradient(135deg, #f57c00 0%, #e65100 100%);
        color: white !important;
    }

    .badge-info {
        background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
        color: white !important;
    }

    /* Make ALL Streamlit text deep colored */
    .stMarkdown p {
        color: #1a237e !important;
        font-weight: 600 !important;
        line-height: 1.8;
    }

    .stMarkdown li {
        color: #1a237e !important;
        font-weight: 600 !important;
        line-height: 1.8;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #0d47a1 !important;
        font-weight: 800 !important;
    }

    .stMarkdown strong {
        color: #01579b !important;
        font-weight: 900 !important;
    }

    /* Input fields */
    .stTextInput label, .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #0d47a1 !important;
        font-weight: 700 !important;
        font-size: 1.05rem;
    }

    .stTextInput>div>div>input,
    .stSelectbox>div>div>select,
    .stNumberInput>div>div>input {
        border: 2px solid #1565c0;
        color: #0d47a1 !important;
        font-weight: 600 !important;
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
        st.error(f"📁 Data files not found: {e}")
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
        st.error(f"🔧 Preprocessing models not found: {e}")
        return None, None, None, None, None

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models with animated progress"""

    models = {
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    results = {}
    trained_models = {}

    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    for i, (name, model) in enumerate(models.items()):
        # Show current model being trained
        status_placeholder.markdown(f"""
        <div class="info-card">
            <h3>🔄 Training: {name}</h3>
            <p>Model {i+1} of {len(models)}</p>
        </div>
        """, unsafe_allow_html=True)

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Metrics
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
        time.sleep(0.1)  # Small delay for visual effect

    status_placeholder.markdown("""
    <div class="alert alert-success">
        <h3>✅ Training Complete!</h3>
        <p>All models have been successfully trained and evaluated.</p>
    </div>
    """, unsafe_allow_html=True)

    time.sleep(1)
    progress_bar.empty()
    status_placeholder.empty()

    # Champion model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = trained_models[best_model_name]

    return trained_models, results, best_model_name, best_model

def create_interactive_form():
    """Create highly interactive survey form with better UX"""
    st.markdown('<div class="sub-header">📋 Interactive Mental Health Assessment</div>', unsafe_allow_html=True)

    # Progress tracking
    if 'form_progress' not in st.session_state:
        st.session_state.form_progress = 0

    # Create organized tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Personal Details",
        "💼 Employment Info", 
        "🏢 Workplace Culture",
        "💭 Attitudes & Views"
    ])

    with tab1:
        st.markdown("""
        <div class="info-card">
            <h3>Tell us about yourself</h3>
            <p>Your demographic information helps us provide more accurate predictions.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider(
                "🎂 Age",
                min_value=18,
                max_value=100,
                value=30,
                help="Select your current age"
            )

            gender = st.radio(
                "⚥ Gender",
                ["Male", "Female", "Other"],
                help="Your gender identity"
            )

            country = st.selectbox(
                "🌍 Country",
                [
                    "United States", "Canada", "United Kingdom", "Germany",
                    "Netherlands", "Australia", "France", "India", "Other"
                ],
                help="Country of residence"
            )

        with col2:
            self_employed = st.radio(
                "💼 Employment Type",
                ["Self-employed", "Company Employee"],
                help="Are you self-employed or working for a company?"
            )
            self_employed = "Yes" if self_employed == "Self-employed" else "No"

            family_history = st.select_slider(
                "👨‍👩‍👧‍👦 Family Mental Health History",
                options=["No", "Don't know", "Yes"],
                help="Family history of mental health conditions"
            )

            work_interfere = st.select_slider(
                "🧠 Work Interference",
                options=["Never", "Rarely", "Sometimes", "Often"],
                help="How often does mental health affect your work?"
            )

    with tab2:
        st.markdown("""
        <div class="info-card">
            <h3>Your Workplace Environment</h3>
            <p>Information about your work setting and company.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            tech_company = st.radio(
                "💻 Tech Industry",
                ["Yes", "No"],
                help="Do you work in the technology industry?"
            )

            no_employees = st.selectbox(
                "👥 Company Size",
                ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"],
                help="Number of employees in your company"
            )

            remote_work = st.radio(
                "🏠 Work Location",
                ["Remote", "Office"],
                help="Do you work remotely or from an office?"
            )
            remote_work = "Yes" if remote_work == "Remote" else "No"

        with col2:
            benefits = st.select_slider(
                "🏥 Mental Health Benefits",
                options=["No", "Don't know", "Yes"],
                help="Does your employer provide mental health benefits?"
            )

            wellness_program = st.select_slider(
                "🧘 Wellness Programs",
                options=["No", "Don't know", "Yes"],
                help="Mental health discussion in wellness programs"
            )

            seek_help = st.select_slider(
                "📚 Support Resources",
                options=["No", "Don't know", "Yes"],
                help="Availability of mental health resources"
            )

    with tab3:
        st.markdown("""
        <div class="info-card">
            <h3>Work Environment & Support</h3>
            <p>How your workplace handles mental health matters.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            anonymity = st.select_slider(
                "🔒 Privacy Protection",
                options=["No", "Don't know", "Yes"],
                help="Is anonymity protected when seeking help?"
            )

            leave = st.select_slider(
                "🏖️ Medical Leave Ease",
                options=["Very difficult", "Somewhat difficult", "Don't know", "Somewhat easy", "Very easy"],
                help="How easy is it to take mental health leave?"
            )

        with col2:
            care_options = st.select_slider(
                "ℹ️ Care Options Awareness",
                options=["No", "Not sure", "Yes"],
                help="Do you know available care options?"
            )

            obs_consequence = st.radio(
                "👀 Observed Negative Impact",
                ["No", "Yes"],
                help="Seen negative consequences for colleagues?"
            )

    with tab4:
        st.markdown("""
        <div class="info-card">
            <h3>Your Perceptions & Attitudes</h3>
            <p>How you feel about discussing mental health at work.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            mental_health_consequence = st.select_slider(
                "⚠️ Fear of Consequences",
                options=["No", "Maybe", "Yes"],
                help="Would discussing mental health have negative effects?"
            )

            coworkers = st.select_slider(
                "👥 Coworker Comfort",
                options=["No", "Some of them", "Yes"],
                help="Would you discuss mental health with coworkers?"
            )

            supervisor = st.select_slider(
                "👔 Supervisor Comfort",
                options=["No", "Some of them", "Yes"],
                help="Would you discuss with your supervisor?"
            )

        with col2:
            mental_health_interview = st.select_slider(
                "💼 Interview Disclosure",
                options=["No", "Maybe", "Yes"],
                help="Would you mention it in a job interview?"
            )

            mental_vs_physical = st.select_slider(
                "⚖️ Mental vs Physical Health",
                options=["No", "Don't know", "Yes"],
                help="Does employer treat them equally?"
            )

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
        df = pd.DataFrame([user_input])

        # Encode using label encoders
        for col in df.columns:
            if col in label_encoders:
                try:
                    if df[col].iloc[0] not in label_encoders[col].classes_:
                        st.error(f"❌ Error encoding {col}: '{df[col].iloc[0]}' not in training data")
                        return None
                    df[col] = label_encoders[col].transform(df[col])
                except ValueError as e:
                    st.error(f"❌ Error: {e}")
                    return None

        # Create derived features
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

        # Align features
        model_features_df = pd.DataFrame(0, index=[0], columns=selected_features)
        for feature in selected_features:
            if feature in df.columns:
                model_features_df[feature] = df[feature].values[0]

        # Scale
        df_scaled = scaler.transform(model_features_df)
        return df_scaled
    except Exception as e:
        st.error(f"❌ Preprocessing error: {e}")
        return None

def display_prediction_results(prediction, treatment_prob, no_treatment_prob, best_model_name, model_accuracy, results):
    """Display beautiful animated prediction results"""

    st.markdown("---")
    st.markdown('<div class="sub-header">🎯 Your Prediction Results</div>', unsafe_allow_html=True)

    # Animated reveal
    with st.spinner("🔮 Analyzing your responses..."):
        time.sleep(1.5)

    # Main prediction
    prediction_class = "high-risk" if treatment_prob > 0.6 else "low-risk"
    prediction_text = "✅ Likely to Seek Treatment" if prediction == 1 else "📊 Less Likely to Seek Treatment"
    confidence = max(treatment_prob, no_treatment_prob)

    st.markdown(f"""
    <div class="prediction-box {prediction_class} glow-card">
        <h2>{prediction_text}</h2>
        <p style="font-size: 1.3rem; margin: 1rem 0;">
            <strong>Confidence:</strong> {confidence:.1%}
        </p>
        <p style="font-size: 1.1rem;">
            <span class="badge badge-info">{best_model_name}</span>
            <span class="badge badge-success">{model_accuracy:.1%} Accuracy</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Probability visualization
    st.markdown('<div class="sub-header">📊 Detailed Probability Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=treatment_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Treatment Likelihood (%)", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 40], 'color': "#c8e6c9"},
                    {'range': [40, 70], 'color': "#fff9c4"},
                    {'range': [70, 100], 'color': "#ffccbc"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Probability comparison
        fig = go.Figure(data=[
            go.Bar(
                name='Probability',
                x=['Will Seek Treatment', 'Won\'t Seek Treatment'],
                y=[treatment_prob, no_treatment_prob],
                marker=dict(
                    color=[treatment_prob, no_treatment_prob],
                    colorscale='RdYlGn',
                    line=dict(color='#667eea', width=2)
                ),
                text=[f'{treatment_prob:.1%}', f'{no_treatment_prob:.1%}'],
                textposition='auto',
                textfont=dict(size=16, color='white', family='Inter')
            )
        ])
        fig.update_layout(
            title="Outcome Probabilities",
            yaxis_title="Probability",
            yaxis=dict(tickformat='.0%'),
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # Model performance metrics
    st.markdown('<div class="sub-header">📈 Model Performance Metrics</div>', unsafe_allow_html=True)

    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

    metrics = [
        ("Precision", results[best_model_name]['precision'], "🎯"),
        ("Recall", results[best_model_name]['recall'], "📊"),
        ("F1 Score", results[best_model_name]['f1'], "⭐"),
        ("ROC AUC", results[best_model_name]['roc_auc'], "📈")
    ]

    for col, (metric_name, value, emoji) in zip([metrics_col1, metrics_col2, metrics_col3, metrics_col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card float-element">
                <div style="font-size: 2.5rem; text-align: center;">{emoji}</div>
                <div class="stat-number" style="font-size: 2rem; text-align: center;">{value:.1%}</div>
                <div class="stat-label" style="text-align: center; color: white;">{metric_name}</div>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Animated header
    st.markdown('<div class="main-header">🧠 Mental Health AI Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Advanced Machine Learning for Mental Health Awareness</div>', unsafe_allow_html=True)

    # Load data
    train_data, test_data, complete_data = load_data()
    if train_data is None:
        st.stop()

    # Load preprocessing
    scaler, selector, label_encoders, target_encoder, selected_features = load_preprocessing_models()
    if scaler is None:
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        st.markdown("---")

        page = st.radio(
            "Select Page",
            [
                "🏠 Prediction",
                "📊 Data Insights",
                "🤖 Model Performance",
                "💡 Resources"
            ],
            key="main_nav"
        )


    if page == "🏠 Prediction":
        # Welcome card
        st.markdown("""
        <div class="info-card">
            <h2>👋 Welcome to Mental Health AI Predictor!</h2>
            <p style="font-size: 1.1rem;">
                This AI-powered tool uses machine learning to predict mental health treatment seeking patterns.
                Complete the interactive survey below to receive personalized insights.
            </p>
            <div style="margin-top: 1rem;">
                <span class="badge badge-info">9 ML Models</span>
                <span class="badge badge-success">70%+ Accuracy</span>
                <span class="badge badge-warning">Real-time Analysis</span>  
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Form
        user_input = create_interactive_form()

        # Predict button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_btn = st.button("🔮 Get AI Prediction", use_container_width=True)

        if predict_btn:
            # Preprocess
            processed_input = preprocess_user_input(user_input, label_encoders, scaler, selector, selected_features)

            if processed_input is None:
                st.error("❌ Failed to process input. Please check your responses.")
                st.stop()

            # Train
            X_train = train_data.drop('target', axis=1)
            y_train = train_data['target']
            X_test = test_data.drop('target', axis=1)
            y_test = test_data['target']

            trained_models, results, best_model_name, best_model = train_models(X_train, y_train, X_test, y_test)

            # Predict
            prediction_proba = best_model.predict_proba(processed_input)[0]
            prediction = best_model.predict(processed_input)[0]

            treatment_prob = prediction_proba[1]
            no_treatment_prob = prediction_proba[0]

            # Display results
            display_prediction_results(
                prediction, treatment_prob, no_treatment_prob,
                best_model_name, results[best_model_name]['accuracy'], results
            )

            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                st.markdown('<div class="sub-header">🔍 Key Influencing Factors</div>', unsafe_allow_html=True)

                feature_importance = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)

                fig = px.bar(
                    feature_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Viridis',
                    title="Top 10 Most Important Features"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            # Recommendations
            st.markdown('<div class="sub-header">💡 Personalized Recommendations</div>', unsafe_allow_html=True)

            if prediction == 1:
                st.markdown("""
                <div class="alert alert-success">
                    <h3>✅ Positive Mental Health Awareness</h3>
                    <p><strong>Great news!</strong> Your profile suggests you're likely to seek support when needed.</p>
                    <ul>
                        <li>🎯 Continue prioritizing mental wellness</li>
                        <li>🏥 Utilize available workplace resources</li>
                        <li>💬 Maintain open communication</li>
                        <li>🌟 Help reduce stigma by sharing experiences</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert alert-warning">
                    <h3>⚠️ Mental Health Support Opportunity</h3>
                    <p>You might face barriers to seeking support. Let's address them:</p>
                    <ul>
                        <li>🔍 Explore available mental health resources</li>
                        <li>💬 Talk to trusted friends or HR</li>
                        <li>📚 Educate yourself about mental health</li>
                        <li>🆘 Know crisis support: 988 (Suicide & Crisis Lifeline)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Profile summary
            st.markdown('<div class="sub-header">👤 Your Profile Summary</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                <div class="info-card">
                    <h4>📊 Personal Information</h4>
                    <ul>
                        <li><strong>Age:</strong> {user_input['Age']} years</li>
                        <li><strong>Gender:</strong> {user_input['Gender']}</li>
                        <li><strong>Location:</strong> {user_input['Country']}</li>
                        <li><strong>Employment:</strong> {"Self-employed" if user_input['self_employed'] == 'Yes' else "Company Employee"}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="info-card">
                    <h4>🏢 Workplace Factors</h4>
                    <ul>
                        <li><strong>Industry:</strong> {"Technology" if user_input['tech_company'] == 'Yes' else "Non-Tech"}</li>
                        <li><strong>Size:</strong> {user_input['no_employees']} employees</li>
                        <li><strong>Benefits:</strong> {user_input['benefits']}</li>
                        <li><strong>Remote:</strong> {user_input['remote_work']}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Disclaimer
            st.markdown("""
            <div class="alert alert-info">
                <h4>⚠️ Important Disclaimer</h4>
                <p>This is an AI prediction tool for informational purposes only. It does NOT replace professional medical advice.
                If experiencing mental health concerns, please consult a qualified healthcare provider.</p>
            </div>
            """, unsafe_allow_html=True)

    elif page == "📊 Data Insights":
        st.markdown('<div class="sub-header">📊 Dataset Analysis & Insights</div>', unsafe_allow_html=True)

        try:
            original_data = pd.read_csv('data/processed/cleaned_mental_health_data.csv')

            # Key stats
            st.markdown("### 📈 Key Statistics")

            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

            stats = [
                ("Total Responses", len(original_data), "👥"),
                ("Treatment Rate", f"{original_data['treatment'].value_counts(normalize=True).get('Yes', 0) * 100:.1f}%", "🏥"),
                ("Avg Age", f"{original_data['Age'].mean():.1f}", "🎂"),
                ("Tech Workers", f"{original_data['tech_company'].value_counts(normalize=True).get('Yes', 0) * 100:.1f}%", "💻")
            ]

            for col, (label, value, emoji) in zip([stat_col1, stat_col2, stat_col3, stat_col4], stats):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 2.5rem; text-align: center;">{emoji}</div>
                        <div class="stat-number" style="text-align: center;">{value}</div>
                        <div class="stat-label" style="text-align: center; color: white;">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Visualizations
            st.markdown("### 🎨 Interactive Visualizations")

            viz_tabs = st.tabs(["📊 Treatment", "👥 Demographics", "💼 Workplace"])

            with viz_tabs[0]:
                col1, col2 = st.columns(2)

                with col1:
                    treatment_counts = original_data['treatment'].value_counts()
                    fig = px.pie(
                        values=treatment_counts.values,
                        names=treatment_counts.index,
                        title="Treatment Distribution",
                        hole=0.4,
                        color_discrete_sequence=['#667eea', '#ff7675']
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.bar(
                        x=treatment_counts.index,
                        y=treatment_counts.values,
                        title="Treatment Counts",
                        color=treatment_counts.index,
                        color_discrete_sequence=['#667eea', '#ff7675']
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            with viz_tabs[1]:
                fig = px.box(
                    original_data,
                    x='treatment',
                    y='Age',
                    title="Age Distribution by Treatment",
                    color='treatment',
                    color_discrete_sequence=['#667eea', '#ff7675']
                )
                st.plotly_chart(fig, use_container_width=True)

            with viz_tabs[2]:
                work_treatment = pd.crosstab(
                    original_data['work_interfere'],
                    original_data['treatment'],
                    normalize='index'
                ) * 100
                fig = px.bar(
                    work_treatment,
                    title="Treatment Rate by Work Interference (%)",
                    color_discrete_sequence=['#667eea', '#ff7675']
                )
                st.plotly_chart(fig, use_container_width=True)

        except FileNotFoundError:
            st.error("❌ Data file not found")

    elif page == "🤖 Model Performance":
        st.markdown('<div class="sub-header">🤖 AI Model Performance</div>', unsafe_allow_html=True)

        X_train = train_data.drop('target', axis=1)
        y_train = train_data['target']
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']

        if st.button("🚀 Train & Compare Models", use_container_width=True):
            trained_models, results, best_model_name, best_model = train_models(X_train, y_train, X_test, y_test)

            # Champion announcement
            st.balloons()
            st.markdown(f"""
            <div class="alert alert-success pulse-element">
                <h2>🏆 Champion Model: {best_model_name}</h2>
                <p style="font-size: 1.2rem;">
                    <strong>Accuracy:</strong> {results[best_model_name]['accuracy']:.2%}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Results table
            results_df = pd.DataFrame(results).T.round(4).sort_values('accuracy', ascending=False)
            st.dataframe(results_df, use_container_width=True)

            # Comparison charts
            tab1, tab2 = st.tabs(["📊 Accuracy", "📈 All Metrics"])

            with tab1:
                fig = px.bar(
                    x=results_df.index,
                    y=results_df['accuracy'],
                    color=results_df['accuracy'],
                    color_continuous_scale='Viridis',
                    title="Model Accuracy Comparison"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                fig = go.Figure()
                metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
                colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']

                for metric, color in zip(metrics_to_plot, colors):
                    fig.add_trace(go.Bar(
                        name=metric.capitalize(),
                        x=results_df.index,
                        y=results_df[metric],
                        marker_color=color
                    ))

                fig.update_layout(
                    title="Comprehensive Model Comparison",
                    barmode='group',
                    height=500
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

    elif page == "💡 Resources":
        st.markdown('<div class="sub-header">💡 Mental Health Resources</div>', unsafe_allow_html=True)

        resource_tabs = st.tabs(["🆘 Crisis Support", "💻 Apps & Tools", "📚 Learn More"])

        with resource_tabs[0]:
            st.markdown("""
            <div class="alert alert-danger">
                <h3>🆘 24/7 Crisis Support</h3>
                <ul>
                    <li><strong>988:</strong> Suicide & Crisis Lifeline</li>
                    <li><strong>Text HOME to 741741:</strong> Crisis Text Line</li>
                    <li><strong>1-800-662-4357:</strong> SAMHSA Helpline</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with resource_tabs[1]:
            st.markdown("""
            <div class="info-card">
                <h3>💻 Digital Mental Health Tools</h3>
                <ul>
                    <li><strong>BetterHelp:</strong> Online therapy platform</li>
                    <li><strong>Headspace:</strong> Meditation & mindfulness</li>
                    <li><strong>Calm:</strong> Sleep & relaxation</li>
                    <li><strong>Sanvello:</strong> CBT-based mental health</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with resource_tabs[2]:
            st.markdown("""
            <div class="info-card">
                <h3>📚 Educational Resources</h3>
                <ul>
                    <li><strong>NIMH:</strong> National Institute of Mental Health</li>
                    <li><strong>NAMI:</strong> Mental health education & support</li>
                    <li><strong>WHO:</strong> Global mental health resources</li>
                    <li><strong>Mental Health America:</strong> Screening tools</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p><strong>Made with ❤️ for Mental Health Awareness</strong></p>
        <p><small>© 2025 Mental Health AI Predictor | Educational Tool Only | made by PrabalRoy_7884</small></p> 
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
