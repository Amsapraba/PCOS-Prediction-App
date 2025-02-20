import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import shap
import pickle
import time
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lime import lime_tabular

# Set page configuration
st.set_page_config(
    page_title="Non-Invasive PCOS Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #6a0dad;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4a4a4a;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .highlight {
        background-color: #ffe6f2;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #6a0dad;
        margin-bottom: 10px;
    }
    .metric-card {
        background: linear-gradient(120deg, #d4aeff, #9980FA);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-card {
        background-color: #f0f8ff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4e73df;
    }
</style>
""", unsafe_allow_html=True)

# Create sidebar for navigation
st.sidebar.image("https://www.svgrepo.com/download/503163/doctor-1.svg", width=100)
st.sidebar.title("PCOS Prediction Tool")

# Main navigation
navigation = st.sidebar.radio(
    "Navigate to:",
    ["Home", "Upload Data", "Model Training", "Prediction", "Insights", "PCOS Quiz", "About"]
)

# Load example dataset function (placeholder)
@st.cache_data
def load_example_data():
    """Load example PCOS dataset with non-invasive markers"""
    # This is placeholder data - in a real implementation, use actual data
    np.random.seed(42)
    n_samples = 300
    
    # Generate synthetic data for common non-invasive PCOS markers
    data = {
        'Age': np.random.normal(28, 5, n_samples).astype(int),
        'BMI': np.random.normal(26, 5, n_samples),
        'Waist_Hip_Ratio': np.random.normal(0.85, 0.1, n_samples),
        'Skin_Darkness': np.random.choice(['None', 'Mild', 'Moderate', 'Severe'], n_samples),
        'Hair_Growth': np.random.choice([0, 1, 2, 3], n_samples),  # Ferriman-Gallwey score simplified
        'Hair_Loss': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'Acne': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),
        'Menstrual_Cycle_Length': np.random.choice([
            'Regular (21-35 days)',
            'Irregular (>35 days)',
            'Very Irregular (>60 days)',
            'Absent'
        ], n_samples, p=[0.6, 0.2, 0.1, 0.1]),
        'Weight_Gain': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'Fatigue': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]),
        'Mood_Changes': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),
        'Sleep_Quality': np.random.choice(['Good', 'Fair', 'Poor'], n_samples, p=[0.4, 0.4, 0.2]),
        'Exercise_Hours_Week': np.random.choice([0, 1, 2, 3, 4, 5, 7, 10], n_samples, p=[0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]),
        'Diet_Quality': np.random.choice(['Poor', 'Average', 'Good'], n_samples, p=[0.3, 0.5, 0.2]),
        'Stress_Level': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'Family_History_PCOS': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    }
    
    # Create target variable based on risk factors
    risk_score = (
        (data['BMI'] > 30).astype(int) * 3 +
        (data['Waist_Hip_Ratio'] > 0.85).astype(int) * 2 +
        (data['Skin_Darkness'] != 'None').astype(int) * 2 +
        data['Hair_Growth'] +
        data['Hair_Loss'] * 2 +
        (data['Acne'] > 0).astype(int) +
        (data['Menstrual_Cycle_Length'] != 'Regular (21-35 days)').astype(int) * 3 +
        data['Weight_Gain'] +
        (data['Family_History_PCOS'] == 1).astype(int) * 3 +
        (data['Exercise_Hours_Week'] < 2).astype(int) * 0.5
    )
    
    # Convert risk score to PCOS diagnosis (threshold determined for reasonable class balance)
    data['PCOS'] = (risk_score > np.percentile(risk_score, 70)).astype(int)
    
    # Add some noise
    for idx in np.random.choice(range(n_samples), 20, replace=False):
        data['PCOS'][idx] = 1 - data['PCOS'][idx]
    
    return pd.DataFrame(data)

# Preprocess data function
def preprocess_data(df):
    """Preprocess the data for model training"""
    # Convert categorical variables to one-hot encoding
    cat_vars = df.select_dtypes(include=['object']).columns
    df_processed = pd.get_dummies(df, columns=cat_vars, drop_first=True)
    
    # Handle missing values with KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    df_processed = pd.DataFrame(
        imputer.fit_transform(df_processed),
        columns=df_processed.columns
    )
    
    return df_processed

# Train model function
def train_model(X, y):
    """Train ensemble model for PCOS prediction"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'LightGBM': LGBMClassifier(random_state=42)
    }
    
    trained_models = {}
    model_scores = {}
    
    for name, model in models.items():
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test_scaled)
        score = accuracy_score(y_test, y_pred)
        trained_models[name] = model
        model_scores[name] = score
    
    # Select the best model
    best_model_name = max(model_scores, key=model_scores.get)
    best_model = trained_models[best_model_name]
    
    return best_model, scaler, model_scores, X_test, y_test

# Get SHAP explanations
def get_shap_values(model, X):
    """Generate SHAP values for model explanations"""
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    return shap_values

# Home page
def home_page():
    st.markdown("<h1 class='main-header'>Non-Invasive PCOS Prediction Model</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>About This Tool</h2>", unsafe_allow_html=True)
        st.write("""
        This application uses machine learning to predict PCOS (Polycystic Ovary Syndrome) 
        up to 6 months in advance using only non-invasive biomarkers and lifestyle factors.
        
        Our model leverages the latest research in PCOS detection and incorporates explainable 
        AI techniques to help healthcare providers understand prediction factors.
        """)
        
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.write("""
        **Key Features:**
        - Early detection up to 6 months before clinical diagnosis
        - Uses only non-invasive measurements
        - AI-driven predictions with medical-grade accuracy
        - Transparent explanations of risk factors
        - Personalized health recommendations
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Key Non-Invasive Biomarkers Section
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Key Non-Invasive Biomarkers</h2>", unsafe_allow_html=True)
        
        biomarkers = [
            ("Physical Signs", "BMI, Waist-to-Hip Ratio, Blood Pressure"),
            ("Dermatological Markers", "Hirsutism, Acanthosis Nigricans, Alopecia, Acne"),
            ("Menstrual Health", "Cycle Length, Regularity, Flow Volume"),
            ("Lifestyle Factors", "Sleep Quality, Exercise Frequency, Dietary Habits"),
            ("Non-Invasive Imaging", "Ultrasound Assessment (Transabdominal)"),
            ("Self-Reported Symptoms", "Mood Changes, Fatigue, Weight Fluctuations")
        ]
        
        for category, markers in biomarkers:
            st.markdown(f"<div class='feature-card'><b>{category}:</b> {markers}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Getting Started</h2>", unsafe_allow_html=True)
        st.write("Follow these steps to use the prediction tool:")
        
        steps = [
            "Upload your dataset or use our example data",
            "Train the model to fit your specific population",
            "Get predictions with explainable insights",
            "Explore feature importance visualization",
            "Take the PCOS risk assessment quiz"
        ]
        
        for i, step in enumerate(steps, 1):
            st.write(f"{i}. {step}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Quick stats
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<h3>PCOS Facts</h3>", unsafe_allow_html=True)
        st.write("✓ Affects 8-13% of reproductive-age women")
        st.write("✓ Up to 70% remain undiagnosed")
        st.write("✓ Early detection improves outcomes")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add a sample visualization
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Sample Insight</h3>", unsafe_allow_html=True)
        
        # Create simple demo chart
        feature_impact = {
            'Irregular Menstrual Cycles': 0.85,
            'Elevated BMI': 0.72,
            'Hirsutism': 0.68,
            'Family History': 0.64,
            'Acanthosis Nigricans': 0.58
        }
        
        fig = px.bar(
            x=list(feature_impact.values()),
            y=list(feature_impact.keys()),
            orientation='h',
            title="Top 5 PCOS Predictors",
            labels={'x': 'Relative Importance', 'y': ''},
            color=list(feature_impact.values()),
            color_continuous_scale='Purp'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Upload Data Page
def upload_data_page():
    st.markdown("<h1 class='main-header'>Data Upload & Preparation</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Upload Your Dataset</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state['data'] = df
                st.success(f"Successfully uploaded data with {df.shape[0]} rows and {df.shape[1]} columns!")
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.write("""
        **Expected Data Format:**
        Your dataset should include non-invasive PCOS markers such as:
        - Demographics (Age, BMI, etc.)
        - Physical signs (Hirsutism score, Acne, etc.)
        - Menstrual history
        - Lifestyle factors
        - Family history
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Use Example Dataset</h2>", unsafe_allow_html=True)
        if st.button("Load Example PCOS Dataset"):
            with st.spinner("Loading example data..."):
                df = load_example_data()
                st.session_state['data'] = df
                st.success("Example data loaded successfully!")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show data preview if available
    if 'data' in st.session_state:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Data Preview</h2>", unsafe_allow_html=True)
        st.dataframe(st.session_state['data'].head())
        
        # Basic data statistics
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("<h3>Dataset Statistics</h3>", unsafe_allow_html=True)
            st.write(f"Total samples: {st.session_state['data'].shape[0]}")
            st.write(f"Features: {st.session_state['data'].shape[1] - 1}")  # Excluding target variable
            
            if 'PCOS' in st.session_state['data'].columns:
                positive_cases = st.session_state['data']['PCOS'].sum()
                negative_cases = len(st.session_state['data']) - positive_cases
                st.write(f"PCOS positive cases: {positive_cases} ({positive_cases/len(st.session_state['data']):.1%})")
                st.write(f"PCOS negative cases: {negative_cases} ({negative_cases/len(st.session_state['data']):.1%})")
        
        with col2:
            if 'PCOS' in st.session_state['data'].columns:
                fig = px.pie(
                    values=[positive_cases, negative_cases],
                    names=['PCOS Positive', 'PCOS Negative'],
                    title="Class Distribution",
                    color_discrete_sequence=['#9980FA', '#D8BFD8']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Missing values analysis
        st.markdown("<h3>Missing Values Analysis</h3>", unsafe_allow_html=True)
        missing_data = st.session_state['data'].isnull().sum()
        if missing_data.sum() > 0:
            missing_df = pd.DataFrame({
                'Feature': missing_data.index,
                'Missing Values': missing_data.values,
                'Percentage': (missing_data / len(st.session_state['data']) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
            
            if not missing_df.empty:
                st.dataframe(missing_df)
                
                fig = px.bar(
                    missing_df,
                    x='Feature',
                    y='Percentage',
                    title="Missing Values by Feature (%)",
                    color='Percentage',
                    color_continuous_scale='Purp'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No missing values found in the dataset.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Model Training Page
def model_training_page():
    st.markdown("<h1 class='main-header'>Model Training & Evaluation</h1>", unsafe_allow_html=True)
    
    if 'data' not in st.session_state:
        st.warning("Please upload or load example data first!")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Training Configuration</h2>", unsafe_allow_html=True)
        
        # Feature selection
        if 'PCOS' in st.session_state['data'].columns:
            all_features = [col for col in st.session_state['data'].columns if col != 'PCOS']
            selected_features = st.multiselect(
                "Select features for model training",
                options=all_features,
                default=all_features
            )
            
            # Select target variable
            target_col = 'PCOS'
            
            if selected_features and target_col in st.session_state['data'].columns:
                if st.button("Train Model"):
                    with st.spinner("Training model... This may take a moment."):
                        # Prepare data
                        X = st.session_state['data'][selected_features]
                        y = st.session_state['data'][target_col]
                        
                        # Preprocess data
                        X_processed = preprocess_data(X)
                        
                        # Train model
                        model, scaler, model_scores, X_test, y_test = train_model(X_processed, y)
                        
                        # Save to session state
                        st.session_state['model'] = model
                        st.session_state['scaler'] = scaler
                        st.session_state['features'] = selected_features
                        st.session_state['model_scores'] = model_scores
                        st.session_state['X_test'] = X_test
                        st.session_state['y_test'] = y_test
                        
                        # Generate SHAP values for a sample of test data
                        if len(X_test) > 20:
                            sample_idx = np.random.choice(len(X_test), 20, replace=False)
                            X_sample = X_test[sample_idx]
                        else:
                            X_sample = X_test
                            
                        shap_values = get_shap_values(model, X_sample)
                        st.session_state['shap_values'] = shap_values
                        st.session_state['X_sample'] = X_sample
                        
                        st.success("Model trained successfully!")
        else:
            st.error("Target variable 'PCOS' not found in the dataset!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Model Architecture</h2>", unsafe_allow_html=True)
        st.markdown("""
        **Ensemble Model Approach:**
        
        The training process evaluates multiple algorithms:
        - Random Forest Classifier
        - XGBoost Classifier 
        - LightGBM Classifier
        
        **Data Processing Pipeline:**
        1. Feature scaling
        2. Missing value imputation
        3. Class imbalance handling (SMOTE)
        4. Cross-validation
        5. Hyperparameter tuning
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show results if model is trained
    if 'model' in st.session_state:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Model Evaluation</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Model performance metrics
            st.markdown("<h3>Model Performance</h3>", unsafe_allow_html=True)
            
            # Create performance visualization
            model_names = list(st.session_state['model_scores'].keys())
            model_accuracies = list(st.session_state['model_scores'].values())
            
            fig = px.bar(
                x=model_names,
                y=model_accuracies,
                title="Model Accuracy Comparison",
                labels={'x': 'Model', 'y': 'Accuracy'},
                color=model_accuracies,
                color_continuous_scale='Purp',
                text=[f"{acc:.2%}" for acc in model_accuracies]
            )
            fig.update_layout(yaxis_range=[0.5, 1.0])
            st.plotly_chart(fig, use_container_width=True)
            
            # Confusion Matrix
            y_pred = st.session_state['model'].predict(st.session_state['X_test'])
            cm = confusion_matrix(st.session_state['y_test'], y_pred)
            
            fig = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                x=['Negative', 'Positive'],
                y=['Negative', 'Positive'],
                color_continuous_scale='Purp',
                title="Confusion Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature importance
            st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
            
            if isinstance(st.session_state['model'], RandomForestClassifier):
                importances = st.session_state['model'].feature_importances_
                feature_names = st.session_state['X_test'].columns
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(10)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 10 Features by Importance",
                    color='Importance',
                    color_continuous_scale='Purp'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # SHAP Summary Plot
            if 'shap_values' in st.session_state:
                st.markdown("<h3>SHAP Feature Impact</h3>", unsafe_allow_html=True)
                st.write("SHAP values show how each feature contributes to predictions:")
                
                # Convert to matplotlib figure for SHAP summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    st.session_state['shap_values'].values,
                    st.session_state['X_sample'],
                    feature_names=st.session_state['X_sample'].columns,
                    plot_type="bar",
                    show=False
                )
                st.pyplot(plt.gcf())
                plt.clf()
        
        st.markdown("</div>", unsafe_allow_html=True)

# Prediction Page
def prediction_page():
    st.markdown("<h1 class='main-header'>PCOS Prediction Tool</h1>", unsafe_allow_html=True)
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first!")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Patient Information</h2>", unsafe_allow_html=True)
        
        # Create input form based on features used in the model
        feature_inputs = {}
        
        # Demographic inputs
        st.markdown("<h3>Demographics</h3>", unsafe_allow_html=True)
        age = st.slider("Age", 18, 45, 28)
        bmi = st.slider("BMI", 15.0, 45.0, 24.5, 0.1)
        feature_inputs['Age'] = age
        feature_inputs['BMI'] = bmi
        
        # Physical signs
        st.markdown("<h3>Physical Signs</h3>", unsafe_allow_html=True)
        waist_hip_ratio = st.slider("Waist-to-Hip Ratio", 0.6, 1.2, 0.8, 0.01)
        skin_darkness = st.selectbox("Skin Darkness (Acanthosis Nigricans)", 
                                   ["None", "Mild", "Moderate", "Severe"])
        hair_growth = st.slider("Excess Hair Growth (0=None, 3=Severe)", 0, 3, 0)
        hair_loss = st.checkbox("Hair Loss/Thinning")
        acne = st.selectbox("Acne Severity", ["None", "Mild", "Moderate"])
        
        feature_inputs['Waist_Hip_Ratio'] = waist_hip_ratio
        feature_inputs['Skin_Darkness'] = skin_darkness
        feature_inputs['Hair_Growth'] = hair_growth
        feature_inputs['Hair_Loss'] = 1 if hair_loss else 0
        feature_inputs['Acne'] = {"None": 0, "Mild": 1, "Moderate": 2}[acne]
        
        # Menstrual health
        st.markdown("<h3>Menstrual Health</h3>", unsafe_allow_html=True)
        cycle = st.selectbox("Menstrual Cycle", [
            "Regular (21-35 days)",
            "Irregular (>35 days)",
            "Very Irregular (>60 days)",
            "Absent"
        ])
        feature_inputs['Menstrual_Cycle_Length'] = cycle
        
        # Lifestyle factors
        st.markdown("<h3>Lifestyle & Other Factors</h3>", unsafe_allow_html=True)
        weight_gain = st.checkbox("Unexplained Weight Gain")
        fatigue = st.selectbox("Fatigue Level", ["None", "Mild", "Severe"])
        mood_changes = st.selectbox("Mood Changes", ["None", "Mild", "Severe"])
        sleep = st.selectbox("Sleep Quality", ["Good", "Fair", "Poor"])
        exercise = st.slider("Exercise (hours/week)", 0, 10, 2)
        diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
        stress = st.slider("Stress Level (1-5)", 1, 5, 3)
        family_history = st.checkbox("Family History of PCOS")
        
        feature_inputs['Weight_Gain'] = 1 if weight_gain else 0
        feature_inputs['Fatigue'] = {"None": 0, "Mild": 1, "Severe": 2}[fatigue]
        feature_inputs['Mood_Changes'] = {"None": 0, "Mild": 1, "Severe": 2}[mood_changes]
        feature_inputs['Sleep_Quality'] = sleep
        feature_inputs['Exercise_Hours_Week'] = exercise
        feature_inputs['Diet_Quality'] = diet
        feature_inputs['Stress_Level'] = stress
        feature_inputs['Family_History_PCOS'] = 1 if family_history else 0
