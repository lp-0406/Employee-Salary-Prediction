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
    
    .input-container h3 {
        color: #4a5568;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
        font-size: 1.5rem;
    }
    
    .stSelectbox > div > div > div {
        background-color: #f7fafc;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stSlider > div > div > div > div {
        background-color: #667eea;
    }
    
    .stSlider > div > div > div > div > div {
        background-color: #4c51bf;
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
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
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
    
    .info-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .batch-section {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
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
            return None, None, "Model file 'best_model.pkl' not found. Please ensure the model file is in the same directory as this app."
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

def load_encoders():
    """Load the saved label encoders"""
    try:
        encoders = joblib.load("label_encoders.pkl")
        return encoders, None
    except Exception as e:
        return None, f"Error loading encoders: {str(e)}"

def validate_input(input_df, expected_columns):
    """Validate input data"""
    try:
        # Check if all expected columns are present
        missing_cols = set(expected_columns) - set(input_df.columns)
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"
        
        # Reorder columns to match expected order
        input_df = input_df[expected_columns]
        return True, input_df
    except Exception as e:
        return False, f"Input validation error: {str(e)}"

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
    st.markdown(f"""
        <div class="error-message">
            <h3>⚠️ Setup Error</h3>
            <p>{model_error or encoder_error}</p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# Display model information
st.markdown('<div class="info-card">', unsafe_allow_html=True)
st.markdown("### 📊 Model Information")
col_info1, col_info2 = st.columns(2)
with col_info1:
    st.info(f"**Model Type**: {type(model).__name__}")
    st.info(f"**Features Used**: {len(expected_columns)}")
with col_info2:
    st.info(f"**Expected Columns**: {', '.join(expected_columns)}")
st.markdown('</div>', unsafe_allow_html=True)

# Input section
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.markdown('<h3>👤 Enter Employee Information</h3>', unsafe_allow_html=True)

# Create input form with exact column matching
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("### 📊 Demographics")
    age = st.slider("📅 Age", 17, 75, 35, help="Employee's age in years")
    
    educational_num = st.selectbox("🎓 Education Level", 
        options=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        index=8,  # Default to 13 (Bachelor's)
        help="Education level: 5=Elementary, 9=HS-grad, 13=Bachelors, 14=Masters, 16=PhD")

with col2:
    st.markdown("### 💼 Work Details")
    
    # Work class selection
    if encoders and 'workclass' in encoders:
        workclass_options = list(encoders['workclass'].classes_)
        workclass_str = st.selectbox("🏢 Work Class", workclass_options, 
                                   help="Type of employer")
        workclass = encoders['workclass'].transform([workclass_str])[0]
    else:
        st.error("Workclass encoder not found")
        st.stop()
    
    # Occupation selection
    if encoders and 'occupation' in encoders:
        occupation_options = list(encoders['occupation'].classes_)
        occupation_str = st.selectbox("💼 Occupation", occupation_options,
                                    help="Job role/occupation")
        occupation = encoders['occupation'].transform([occupation_str])[0]
    else:
        st.error("Occupation encoder not found")
        st.stop()

with col3:
    st.markdown("### 👥 Personal Details")
    
    # Marital status selection
    if encoders and 'marital-status' in encoders:
        marital_options = list(encoders['marital-status'].classes_)
        marital_str = st.selectbox("💑 Marital Status", marital_options,
                                 help="Current marital status")
        marital_status = encoders['marital-status'].transform([marital_str])[0]
    else:
        st.error("Marital status encoder not found")
        st.stop()
    
    # Relationship selection
    if encoders and 'relationship' in encoders:
        relationship_options = list(encoders['relationship'].classes_)
        relationship_str = st.selectbox("👨‍👩‍👧‍👦 Relationship", relationship_options,
                                      help="Relationship within household")
        relationship = encoders['relationship'].transform([relationship_str])[0]
    else:
        st.error("Relationship encoder not found")
        st.stop()

# Additional inputs row
col4, col5 = st.columns([1, 1])

with col4:
    # Race selection
    if encoders and 'race' in encoders:
        race_options = list(encoders['race'].classes_)
        race_str = st.selectbox("🌍 Race", race_options, help="Race/ethnicity")
        race = encoders['race'].transform([race_str])[0]
    else:
        st.error("Race encoder not found")
        st.stop()
    
    # Gender selection
    if encoders and 'gender' in encoders:
        gender_options = list(encoders['gender'].classes_)
        gender_str = st.selectbox("⚧ Gender", gender_options, help="Gender")
        gender = encoders['gender'].transform([gender_str])[0]
    else:
        st.error("Gender encoder not found")
        st.stop()

with col5:
    hours_per_week = st.slider("🕒 Hours per Week", 1, 80, 40, 
                              help="Average hours worked per week")
    capital_gain = st.number_input("💰 Capital Gain", 0, 100000, 0,
                                  help="Capital gains from investments")
    capital_loss = st.number_input("💸 Capital Loss", 0, 10000, 0,
                                  help="Capital losses from investments")

st.markdown('</div>', unsafe_allow_html=True)

# Create input dataframe with EXACT feature order from training
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

# Ensure columns are in the correct order (this is CRITICAL)
try:
    input_data = input_data[expected_columns]
except KeyError as e:
    st.error(f"Column mismatch error: {e}")
    st.write("Expected columns:", expected_columns)
    st.write("Available columns:", list(input_data.columns))
    st.stop()

# Display input preview with user-friendly names
st.markdown('<div class="info-card">', unsafe_allow_html=True)
st.markdown("### 📋 Input Data Preview")
display_df = pd.DataFrame({
    'Age': [age],
    'Work Class': [workclass_str],
    'Education Level': [educational_num],
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
st.markdown('</div>', unsafe_allow_html=True)

# Prediction section
col_predict1, col_predict2, col_predict3 = st.columns([1, 2, 1])
with col_predict2:
    if st.button("🚀 Predict Salary Class", key="predict_btn"):
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Try to get prediction probabilities
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
                    <p><strong>Debug Info:</strong></p>
                    <p>Input shape: {input_data.shape}</p>
                    <p>Expected columns: {expected_columns}</p>
                    <p>Input columns: {list(input_data.columns)}</p>
                </div>
            """, unsafe_allow_html=True)

# Batch prediction section
st.markdown('<div class="batch-section">', unsafe_allow_html=True)
st.markdown("### 📂 Batch Prediction from CSV")
st.markdown("Upload a CSV file with employee data to get predictions for multiple employees at once.")

uploaded_file = st.file_uploader(
    "Choose CSV File", 
    type=["csv"],
    help="Upload a CSV file with columns matching the input format"
)

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        
        st.markdown("### 📑 Uploaded Data Preview")
        st.dataframe(batch_data.head(10), use_container_width=True)
        
        if st.button("🔄 Process Batch Predictions", key="batch_predict"):
            try:
                # Validate batch data
                is_valid, result = validate_input(batch_data, expected_columns)
                
                if not is_valid:
                    st.markdown(f"""
                        <div class="error-message">
                            <h4>❌ Batch Data Error</h4>
                            <p>{result}</p>
                            <p><strong>Expected columns:</strong> {', '.join(expected_columns)}</p>
                            <p><strong>Your CSV columns:</strong> {', '.join(batch_data.columns)}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    batch_data_processed = result
                    predictions = model.predict(batch_data_processed)
                    
                    # Add predictions to the dataframe
                    batch_results = batch_data.copy()
                    batch_results["Predicted_Salary_Class"] = predictions
                    
                    # Try to add confidence scores
                    try:
                        probabilities = model.predict_proba(batch_data_processed)
                        confidence_scores = np.max(probabilities, axis=1) * 100
                        batch_results["Confidence_%"] = confidence_scores.round(1)
                    except:
                        pass
                    
                    st.markdown("### ✅ Batch Prediction Results")
                    st.dataframe(batch_results, use_container_width=True)
                    
                    # Download button
                    csv_data = batch_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name="salary_predictions_results.csv",
                        mime="text/csv",
                        key="download_batch"
                    )
                    
                    # Show summary statistics
                    st.markdown("### 📊 Prediction Summary")
                    summary_col1, summary_col2 = st.columns(2)
                    
                    with summary_col1:
                        high_salary_count = (predictions == '>50K').sum()
                        st.markdown(f"""
                            <div class="metric-card">
                                <h4>💰 High Salary (>50K)</h4>
                                <h2>{high_salary_count}</h2>
                                <p>{(high_salary_count/len(predictions)*100):.1f}% of total</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with summary_col2:
                        low_salary_count = (predictions == '<=50K').sum()
                        st.markdown(f"""
                            <div class="metric-card">
                                <h4>💼 Standard Salary (≤50K)</h4>
                                <h2>{low_salary_count}</h2>
                                <p>{(low_salary_count/len(predictions)*100):.1f}% of total</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.markdown(f"""
                    <div class="error-message">
                        <h4>❌ Batch Processing Error</h4>
                        <p>{str(e)}</p>
                    </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.markdown(f"""
            <div class="error-message">
                <h4>❌ File Reading Error</h4>
                <p>{str(e)}</p>
                <p>Please ensure your CSV file is properly formatted.</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; padding: 2rem; color: rgba(255, 255, 255, 0.7);">
        <p>🚀 Powered by Machine Learning | Built with Streamlit</p>
        <p style="font-size: 0.9rem;">Model trained on Adult Census Income Dataset</p>
    </div>
""", unsafe_allow_html=True)