# pages/1_🪐_Predictor.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Predictor de Exoplanetas", page_icon="🪐", layout="wide")

# --- Funciones de Carga (Caché para eficiencia) ---
@st.cache_resource
def load_artifacts():
    """Carga el modelo, el dataset base y prepara los artefactos necesarios."""
    try:
        model = joblib.load('artifacts/exoplanet_model.joblib')
        df = pd.read_csv('artifacts/final_dataset.csv')
    except FileNotFoundError:
        return None, None, None, None
    
    le = LabelEncoder().fit(df['disposition'])
    df_dummies = pd.get_dummies(df.drop('disposition', axis=1), columns=['mission'])
    model_columns = df_dummies.columns.tolist()
    imputation_values = {
        'disposition_score': df['disposition_score'].median(),
        'signal_to_noise': df['signal_to_noise'].median()
    }
    return model, le, model_columns, imputation_values

model, le, model_columns, imputation_values = load_artifacts()

# --- Interfaz de Usuario ---
st.title("🪐 Módulo de Predicción")
st.write("Introduce las características de un objeto de interés para clasificarlo usando nuestro modelo pre-entrenado.")

if model is None:
    st.error("Error: Archivos del modelo no encontrados. Asegúrate de que la carpeta 'artifacts' con 'exoplanet_model.joblib' y 'final_dataset.csv' existe.")
else:
    params = {}
    
    st.header("Parámetros del Objeto")
    
    # --- Fila 1: Parámetros Globales ---
    st.subheader("✅ Características Globales (Comunes a todas las misiones)")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        params['mission'] = st.selectbox("Misión de Origen", ["Kepler", "K2", "TESS"])
    with col2:
        params['orbital_period'] = st.number_input("Periodo Orbital (días)", min_value=0.0, value=10.5, format="%.4f")
    with col3:
        params['planet_radius_earth'] = st.number_input("Radio del Planeta (Radios 🌎)", min_value=0.0, value=1.6)
    with col4:
        params['planet_temp'] = st.number_input("Temperatura (K)", min_value=0, value=1000)
    with col5:
        params['planet_count_in_system'] = st.number_input("Planetas en Sistema", min_value=1, value=1, step=1)

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        params['transit_depth'] = st.number_input("Profundidad Tránsito (ppm)", min_value=0.0, value=500.0)
    with col2:
        params['transit_duration'] = st.number_input("Duración Tránsito (horas)", min_value=0.0, value=3.5)
    with col3:
        params['impact_parameter'] = st.slider("Parámetro de Impacto", 0.0, 2.0, 0.5, 0.01)
    with col4:
        params['stellar_temperature'] = st.number_input("Temp. Estrella (K)", min_value=2000, value=5778)
    with col5:
        params['stellar_radius'] = st.number_input("Radio Estrella (Radios ☀️)", min_value=0.0, value=1.0)
    with col6:
        params['stellar_mass'] = st.number_input("Masa Estrella (Masas ☀️)", min_value=0.0, value=1.0)
    with col7:
        params['stellar_logg'] = st.number_input("Gravedad Estelar (log g)", min_value=0.0, value=4.4)

    st.divider()

    # --- Fila 2: Características No Comunes ---
    st.subheader("🛰️ Características No Comunes (se habilitan según la misión)")
    if params['mission'] == 'Kepler':
        st.info("Estas características de diagnóstico solo están disponibles para la misión Kepler y mejoran significativamente la predicción.")
        k_col1, k_col2, k_col3, k_col4, k_col5, k_col6 = st.columns(6)
        with k_col1:
            params['disposition_score'] = st.slider("Score de Disposición", 0.0, 1.0, 0.95, 0.01)
        with k_col2:
            params['signal_to_noise'] = st.number_input("Señal-Ruido (SNR)", min_value=0.0, value=50.0)
        with k_col3:
            params['fp_flag_nt'] = st.selectbox("Flag NT", [0, 1], help="Not Transit-Like Flag")
        with k_col4:
            params['fp_flag_ss'] = st.selectbox("Flag SS", [0, 1], help="Stellar Eclipse Flag")
        with k_col5:
            params['fp_flag_co'] = st.selectbox("Flag CO", [0, 1], help="Centroid Offset Flag")
        with k_col6:
            params['fp_flag_ec'] = st.selectbox("Flag EC", [0, 1], help="Ephemeris Contamination Flag")
    else:
        st.warning(f"La misión '{params['mission']}' no proporciona estas características de diagnóstico. Se utilizarán valores neutros para la predicción.")

    st.divider()

    # Botón de predicción centrado
    _, center_col, _ = st.columns([2, 1, 2])
    if center_col.button("Clasificar Objeto", use_container_width=True, type="primary"):
        # Lógica de predicción
        input_data = {}
        for feature in model_columns:
            if feature.startswith('mission_'):
                input_data[feature] = 0
            elif feature in params:
                input_data[feature] = params[feature]
            elif params['mission'] != 'Kepler' and feature in imputation_values:
                input_data[feature] = imputation_values[feature]
            elif params['mission'] != 'Kepler' and feature.startswith('fp_flag_'):
                input_data[feature] = 0
            else:
                input_data[feature] = 0

        mission_column = f"mission_{params['mission']}"
        if mission_column in input_data:
            input_data[mission_column] = 1

        input_df = pd.DataFrame([input_data])[model_columns]

        prediction_encoded = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        prediction_label = le.inverse_transform(prediction_encoded)[0]
        
        # Mostrar Resultado
        st.header("Resultado de la Clasificación")
        col1, col2 = st.columns([1, 2])
        with col1:
            if prediction_label == 'CONFIRMED':
                st.success(f"### {prediction_label}")
                st.markdown("### 🪐")
            elif prediction_label == 'CANDIDATE':
                st.info(f"### {prediction_label}")
                st.markdown("### 🔭")
            else:
                st.error(f"### {prediction_label}")
                st.markdown("### 🌟")
        with col2:
            st.markdown("##### Confianza del Modelo")
            proba_df = pd.DataFrame(prediction_proba, columns=le.classes_, index=['Probabilidad'])
            st.dataframe(proba_df.style.format("{:.2%}"), use_container_width=True)