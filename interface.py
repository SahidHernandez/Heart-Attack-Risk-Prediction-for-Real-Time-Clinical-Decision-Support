import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt


# Page configuration
logo = Image.open("ficLogo.png")
st.set_page_config(page_title="FIC", page_icon=logo, layout="wide")

# Load model configurations
model_configs = {
    'Model 1': {
        'model': joblib.load('Dataset_1/ensemble_RandomForest_dataset1.pkl'),
        'scaler': joblib.load('Dataset_1/model_standardscaler.pkl'),
        'variable': ['Age', 'CK-MB', 'Troponin', 'Gender']
    },
    'Model 2': {
        'model': joblib.load('Dataset_2/ensemble_DecisionTree_dataset2.pkl'),
        'scaler': joblib.load('Dataset_2/model_standardscaler.pkl'),
        'variable': ['exang', 'cp', 'oldpeak', 'thalach', 'ca']
    }
}

# Initialize default model
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = 'Model 1'

st.title("Heart Attack Prediction")
st.write("""
This application is intended to support medical decisions and does not replace expert evaluation.
It uses machine learning models to predict the risk of a heart attack based on different biomarkers:
""")

with st.expander("Model 1", expanded=False):
    st.markdown("""
    - **Age:** Patient's age.
    - **CK-MB:** Marker of heart muscle damage, used to assess heart attacks.
    - **Troponin:** A group of proteins released into the bloodstream when the heart muscle is damaged.
    - **Gender:** Patient's gender.
    """)

with st.expander("Model 2", expanded=False):
    st.markdown("""
    - **Exang:** Exercise-induced angina (0 = No, 1 = Yes). 
    - **Cp:** Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal, 3 = asymptomatic).
    - **Oldpeak:** ST depression induced by exercise relative to rest.
    - **Thalach:** Maximum heart rate achieved.
    - **Ca:** Number of major vessels (0–3) colored by fluoroscopy.
    """)

st.markdown("---")

# Model selection buttons
st.subheader("Select the model for prediction")
col1, col2 = st.columns(2)
with col1:
    if st.button("Model 1"):
        st.session_state['model_name'] = 'Model 1'
with col2:
    if st.button("Model 2"):
        st.session_state['model_name'] = 'Model 2'

model_name = st.session_state['model_name']
st.markdown(f"*Selected model:* {model_name}")

# Get model, scaler, and required variables
config = model_configs[model_name]
model = config['model']
scaler = config['scaler']
vars_required = config['variable']

st.markdown("### Enter patient data")

# Dynamic input fields
values = {}

if 'Age' in vars_required:
    values['Age'] = st.number_input("Age", min_value=0, max_value=120, value=45)

if 'CK-MB' in vars_required:
    ckmb = st.number_input("CK-MB", value=2.86, min_value=0.00, format="%.2f")
    values['CK-MB'] = np.log(ckmb + 1e-10)

if 'Troponin' in vars_required:
    troponin = st.number_input("Troponin", value=0.003, min_value=0.000, format="%.3f", step=0.001)
    values['Troponin'] = np.log(troponin + 1e-10)

if 'Gender' in vars_required:
    gender = st.selectbox("Gender", options=["Male", "Female"])
    values['Gender'] = 1 if gender == "Male" else 0

if 'exang' in vars_required:
    values['exang'] = st.selectbox("Exang", options=[0, 1])

if 'cp' in vars_required:
    values['cp'] = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])

if 'oldpeak' in vars_required:
    values['oldpeak'] = st.number_input("Oldpeak", value=1.0, format="%.2f")

if 'thalach' in vars_required:
    values['thalach'] = st.number_input("Maximum Heart Rate (thalach)", min_value=50, max_value=250, value=150)

if 'ca' in vars_required:
    values['ca'] = st.selectbox("Number of vessels (ca)", options=[0, 1, 2, 3])

# Prediction
if st.button("Predict"):
    # Ensure order of variables
    input_df = pd.DataFrame([[values.get(var, np.nan) for var in vars_required]], columns=vars_required)
    input_df = input_df[scaler.feature_names_in_]
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]

    result = "Positive" if pred == 1 else "Negative"
    color = "red" if pred == 1 else "green"

    st.markdown(f"### Result: <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)

    # Show probabilities if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_scaled)[0]
        colors = ['#58b915', "#e97c34"]  # green and orange

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig, ax = plt.subplots(figsize=(2.5, 2.5), facecolor='none')
            wedges, texts, autotexts = ax.pie(
                proba,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                textprops={'color': 'white', 'weight': 'bold', 'fontsize': 10}
            )
            ax.axis('equal')
            fig.patch.set_alpha(0)
            st.pyplot(fig)

            # Custom legend
            st.markdown("""
                <div style="display: flex; justify-content: center; gap: 20px; margin-top: -10px;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 25px; height: 25px; background-color: #58b915; margin-right: 5px; border-radius: 3px;"></div>
                        <span style="font-size: 2em;">Negative</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 25px; height: 25px; background-color: #e97c34; margin-right: 5px; border-radius: 3px;"></div>
                        <span style="font-size: 2em;">Positive</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# Fixed footer
st.markdown("""
    <style>
    .stApp {
        padding-bottom: 80px;
    }

    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #6c757d;
        text-align: center;
        font-size: 0.75em;
        color: #f1f3f6;
        padding: 10px;
        z-index: 9999;
        box-shadow: 0 -1px 5px rgba(0,0,0,0.1);
    }
    </style>

    <div class="footer">
        Facultad de Ingeniería y Ciencias - Universidad Autónoma de Tamaulipas
    </div>
""", unsafe_allow_html=True)

