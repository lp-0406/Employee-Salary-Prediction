import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# App configuration
st.set_page_config(
    page_title="💼 Employee Salary Classifier", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        font-weight: 300;
        margin: 0;
    }
    
    .input-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
        height: 3.5rem;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4);
    }
    
    .error-message {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def load_model_safely():
    """Load the model with error handling"""
    try:
        model_path = Path("best_model.pkl")
        if model_path.exists():
            model, expected_columns = joblib.load(model_path)
            return model, expected_columns, None
        else:
            return None, None, "Model file 'best_model.pkl' not found."
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

def load_encoders():
    """Load the saved label encoders"""
    try:
        encoders = joblib.load("label_encoders.pkl")
        return encoders, None
    except Exception as e:
        return None, f"Error loading encoders: {str(e)}"

# Load model and encoders
model, expected_columns, model_error = load_model_safely()
encoders, encoder_error = load_encoders()

# Main header
st.markdown("""
    <div class="main-header">
        <h1>💼 Employee Salary Classifier</h1>
        <p>Advanced ML-powered salary prediction system</p>
    </div>
""", unsafe_allow_html=True)

if model_error or encoder_error:
    st.error(f"Setup Error: {model_error or encoder_error}")
    st.stop()

# Input section
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #4a5568;">👤 Enter Employee Information</h3>', unsafe_allow_html=True)

# Create input form
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("### 📊 Demographics")
    age = st.slider("📅 Age", 17, 75, 35)
    educational_num = st.selectbox("🎓 Education Level", 
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        index=8,  # Default to 13 (Bachelor's)
        help="5=Elementary, 9=HS-grad, 13=Bachelors, 14=Masters, 16=PhD")

with col2:
    st.markdown("### 💼 Work Details")
    workclass_options = list(encoders['workclass'].classes_)
    workclass_str = st.selectbox("🏢 Work Class", workclass_options)
    workclass = encoders['workclass'].transform([workclass_str])[0]
    
    occupation_options = list(encoders['occupation'].classes_)
    occupation_str = st.selectbox("💼 Occupation", occupation_options)
    occupation = encoders['occupation'].transform([occupation_str])[0]

with col3:
    st.markdown("### 👥 Personal Details")
    marital_options = list(encoders['marital-status'].classes_)
    marital_str = st.selectbox("💑 Marital Status", marital_options)
    marital_status = encoders['marital-status'].transform([marital_str])[0]
    
    relationship_options = list(encoders['relationship'].classes_)
    relationship_str = st.selectbox("👨‍👩‍👧‍👦 Relationship", relationship_options)
    relationship = encoders['relationship'].transform([relationship_str])[0]

# Additional inputs
col4, col5 = st.columns([1, 1])

with col4:
    race_options = list(encoders['race'].classes_)
    race_str = st.selectbox("🌍 Race", race_options)
    race = encoders['race'].transform([race_str])[0]
    
    gender_options = list(encoders['gender'].classes_)
    gender_str = st.selectbox("⚧ Gender", gender_options)
    gender = encoders['gender'].transform([gender_str])[0]

with col5:
    hours_per_week = st.slider("🕒 Hours per Week", 1, 80, 40)
    capital_gain = st.number_input("💰 Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("💸 Capital Loss", 0, 10000, 0)

st.markdown('</div>', unsafe_allow_html=True)

# Create input dataframe with exact feature order
input_data = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week]
})

# Ensure columns are in the correct order
input_data = input_data[expected_columns]

# Display input preview
st.markdown("### 📋 Input Data Preview")
display_df = pd.DataFrame({
    'Age': [age],
    'Work Class': [workclass_str],
    'Education': [educational_num],
    'Marital Status': [marital_str],
    'Occupation': [occupation_str],
    'Relationship': [relationship_str],
    'Race': [race_str],
    'Gender': [gender_str],
    'Capital Gain': [capital_gain],
    'Capital Loss': [capital_loss],
    'Hours/Week': [hours_per_week]
})
st.dataframe(display_df, use_container_width=True)

# Prediction
col_predict1, col_predict2, col_predict3 = st.columns([1, 2, 1])
with col_predict2:
    if st.button("🚀 Predict Salary Class"):
        try:
            prediction = model.predict(input_data)[0]
            try:
                prob = model.predict_proba(input_data)[0]
                confidence = max(prob) * 100
                st.markdown(f"""
                    <div class="prediction-result">
                        <h2>🎯 Prediction Result</h2>
                        <h3>Salary Class: {prediction}</h3>
                        <p>Confidence: {confidence:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown(f"""
                    <div class="prediction-result">
                        <h2>🎯 Prediction Result</h2>
                        <h3>Salary Class: {prediction}</h3>
                    </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
                <div class="error-message">
                    <h4>❌ Prediction Error</h4>
                    <p>{str(e)}</p>
                </div>
            """, unsafe_allow_html=True)

# Batch prediction section
st.markdown("---")
st.markdown("### 📂 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("📑 Uploaded Preview:")
        st.dataframe(batch_data.head())
        
        if st.button("🔄 Process Batch Predictions"):
            # Ensure columns match expected format
            batch_data_processed = batch_data[expected_columns]
            predictions = model.predict(batch_data_processed)
            
            batch_results = batch_data.copy()
            batch_results["Predicted_Salary_Class"] = predictions
            
            st.write("✅ Prediction Results:")
            st.dataframe(batch_results)
            
            csv = batch_results.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions CSV", csv, "salary_predictions.csv", "text/csv")
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
