import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import matplotlib.pyplot as plt

# Configuración de la página
logo = Image.open("LogoUAT.png")
st.set_page_config(page_title="FIC", page_icon = logo, layout="wide")

# Cargar configuraciones de modelos
config_modelos = {
    'Model 1': {
        'model': joblib.load('1st_dataset_final/ensamble_RandomForest_dataset1.pkl'),
        'scaler': joblib.load('1st_dataset_final/model_standardscaler.pkl'),
        'variable': ['Age', 'CK-MB', 'Troponin', 'Gender']
    },
    'Model 2': {
        'model': joblib.load('2nd_dataset_final/ensamble_DecisionTree_dataset2.pkl'),
        'scaler': joblib.load('2nd_dataset_final/model_standardscaler.pkl'),
        'variable': ['exang', 'cp', 'oldpeak', 'thalach', 'ca']
    }
}

# Inicializar modelo por defecto
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = 'Model 1'

st.title("Heart Attack Classifier")
st.write("""
This application is only for medical decision support and does not replace expert evaluation.
Machine learning techniques are used to predict the risk of a heart attack in patients. For this, we use two models with different biomarkers:
""")

with st.expander("Model 1", expanded=False):
    st.markdown("""
    - **Age:** Patient´s age.
    - **CK-MB:** Creatine Kinase MB, it serves as a marker of damage to the heart muscle and is used to diagnose or evaluate conditions such as myocardial infarction (heart attack).
    - **Troponin:** It is a group of proteins present in the heart muscles. Troponin is released into the bloodstream when there is damage to the heart muscle.
    - **Gender:** Patient´s gender.
    
    """)
    
with st.expander("Model 2", expanded=False):
    st.markdown("""
    - **Exang:** Exercise-induced angina (0 = False, 1 = True). 
    - **Cp:** Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal, 3 = asymptomatic).
    - **Oldpeak:** ST depression induced by exercise ST segment.
    - **Thalach:** Maximum heart rate archived.
    - **Ca:** Number of major vassels (0-3) colored by fluoroscopy.
    
    """)

st.markdown("---")

# Selección de modelo con botones
st.subheader("Select the model for prediction")
col1, col2 = st.columns(2)
with col1:
    if st.button("Model 1"):
        st.session_state['modelo_nombre'] = 'Model 1'
with col2:
    if st.button("Model 2"):
        st.session_state['modelo_nombre'] = 'Model 2'

modelo_nombre = st.session_state['modelo_nombre']
st.markdown(f"*Selected model:* {modelo_nombre}")

# Obtener modelo, scaler y variables
config = config_modelos[modelo_nombre]
model = config['model']
scaler = config['scaler']
vars_required = config['variable']

st.markdown("### Enter the patient's data")

# Entradas condicionales
values = {}

if 'Age' in vars_required:
    valores['Age'] = st.number_input("Age", min_value=0, max_value=120, value=45)

if 'CK-MB' in vars_required:
    ckmb = st.number_input("CK-MB", value=2.86, min_value=0.00, format="%.2f")
    valores['CK-MB'] = np.log(ckmb + 1e-10)

if 'Troponin' in vars_required:
    troponina = st.number_input("Troponin", value=0.003, min_value=0.000, format="%.3f", step=0.001)
    valores['Troponin'] = np.log(troponina + 1e-10)

if 'Gender' in vars_required:
    valores['Gender'] = st.selectbox("Gender", options=["Male", "Female"])
    # Convertir a valor numérico si el modelo lo requiere
    valores['Gender'] = 1 if valores['Gender'] == "Male" else 0

if 'exang' in vars_required:
    valores['exang'] = st.selectbox("exang", options=[0, 1])

if 'cp' in vars_required:
    valores['cp'] = st.selectbox("cp", options=[0, 1, 2, 3])

if 'oldpeak' in vars_required:
    valores['oldpeak'] = st.number_input("oldpeak", value=1.0, format="%.2f")

if 'thalach' in vars_required:
    valores['thalach'] = st.number_input("thalach", min_value=50, max_value=250, value=150)

if 'ca' in vars_required:
    valores['ca'] = st.selectbox("ca", options=[0, 1, 2, 3])

# Procesar predicción al hacer clic
if st.button("Classify"):
    # Asegura el orden y la presencia de todas las variables requeridas
    input = pd.DataFrame([[values.get(var, np.nan) for var in vars_required:]], columns=vars_required:)
    input = entrada[scaler.feature_names_in_]
    input_scaled = scaler.transform(input)
    pred = modelo.predict(input_scaled)[0]

    result = "Positive" if pred == 1 else "Negative"
    color = "red" if pred == 1 else "green"

    st.markdown(f"### Result: <span style='color:{color}'>{resultado}</span>", unsafe_allow_html=True)

    # Mostrar probabilidades si están disponibles
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(entrada_scaled)[0]
        colors = ['#58b915', "#e97c34"]  # verde y naranja

        # Usar columnas para centrar la gráfica (3 columnas)
        col1, col2, col3 = st.columns([1, 2, 1])  # col2 es la del centro
        with col2:
            fig, ax = plt.subplots(figsize=(2.5, 2.5), facecolor='none')  # Tamaño compacto
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

            # Leyenda personalizada debajo de la gráfica
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





#Pie de página fijo 
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
