import streamlit as st
import pandas as pd
import joblib

# Load the model and expected columns
model, expected_columns = joblib.load("best_model.pkl")

# App configuration
st.set_page_config(page_title="💼 Employee Salary Classifier", layout="centered")

# Custom CSS for dark theme
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #e0e0e0;
    }
    .stApp {
        background-color: #1e1e1e;
    }
    .css-1cpxqw2 {
        color: #ffffff;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div > select,
    .stSlider > div > div > div > input {
        background-color: #2e2e2e;
        color: white;
        border-radius: 5px;
    }
    .stMarkdown {
        color: #c7c7c7;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.title("💼 Employee Salary Classifier (Dark Mode)")
st.markdown("This app predicts if an employee earns **>50K or ≤50K** based on profile inputs.")

st.subheader("👤 Enter Employee Information")

# Input widgets in columns
col1, col2 = st.columns(2)

with col1:
    age = st.slider("📅 Age", 18, 65, 30)
    education = st.selectbox("🎓 Education Level", [
        "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
    ])
    experience = st.slider("📊 Years of Experience", 0, 40, 5)

with col2:
    occupation = st.selectbox("💼 Job Role", [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
        "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
        "Protective-serv", "Armed-Forces"
    ])
    hours_per_week = st.slider("🕒 Hours per Week", 1, 80, 40)

# Form the input dataframe
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

st.write("### 📋 Input Preview")
st.dataframe(input_df.style.set_properties(**{'background-color': '#1e1e1e', 'color': 'white'}))

# Prediction
if st.button("🚀 Predict Salary Class"):
    input_df = input_df[expected_columns]
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 **Prediction: {prediction}**")

# Batch prediction
st.markdown("---")
st.subheader("📂 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    st.write("📑 Uploaded Preview:")
    st.dataframe(batch_data.head())
    try:
        batch_data = batch_data[expected_columns]
        batch_data["PredictedClass"] = model.predict(batch_data)
        st.write("✅ Prediction Results:")
        st.dataframe(batch_data)
        csv = batch_data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions CSV", csv, "salary_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")
