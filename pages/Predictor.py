import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys
sys.path.append('..')
from sidebar_config import setup_sidebar

st.set_page_config(page_title="Predictor de Exoplanetas", page_icon="", layout="wide")

# Configurar sidebar consistente
setup_sidebar()

# Font Awesome para iconos
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

# --- Funciones de Carga (Caché para eficiencia) ---
@st.cache_resource
def load_default_artifacts():
    """Carga el modelo por defecto, el dataset base y prepara los artefactos necesarios."""
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

@st.cache_resource
def load_custom_model(model_filename):
    """Carga un modelo personalizado guardado desde Model Workshop."""
    try:
        model = joblib.load(f'artifacts/saved_models/{model_filename}.joblib')
        le = joblib.load(f'artifacts/saved_models/{model_filename}_labelencoder.joblib')
        model_columns = joblib.load(f'artifacts/saved_models/{model_filename}_columns.joblib')

        # Cargar dataset para valores de imputación
        df = pd.read_csv('artifacts/final_dataset.csv')
        imputation_values = {
            'disposition_score': df['disposition_score'].median(),
            'signal_to_noise': df['signal_to_noise'].median()
        }
        return model, le, model_columns, imputation_values
    except Exception as e:
        st.error(f"Error al cargar modelo personalizado: {str(e)}")
        return None, None, None, None

def get_available_models():
    """Obtiene la lista de modelos disponibles."""
    models = [{'name': 'Modelo por Defecto', 'file': 'default'}]

    index_path = 'artifacts/saved_models/models_index.json'
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r') as f:
                saved_models = json.load(f)
            models.extend(saved_models)
        except:
            pass

    return models

@st.cache_data
def get_default_model_metrics():
    """Carga las métricas del modelo predeterminado desde el archivo JSON."""
    try:
        with open('artifacts/default_model_index.json', 'r') as f:
            default_model_data = json.load(f)

        # Obtener el primer (y único) modelo del archivo
        model_info = default_model_data[0]
        metrics = model_info['evaluation_metrics']

        return {
            'algorithm': model_info.get('algorithm', 'Random Forest'),
            'accuracy': metrics.get('accuracy', 0),
            'avg_precision': metrics.get('avg_precision', 0),
            'avg_recall': metrics.get('avg_recall', 0),
            'avg_f1': metrics.get('avg_f1', 0)
        }
    except Exception as e:
        # En caso de error, retornar valores por defecto
        return {
            'algorithm': 'RandomForestClassifier',
            'accuracy': 0.897,
            'avg_precision': 0.901,
            'avg_recall': 0.897,
            'avg_f1': 0.897
        }

def prepare_input_data(params, model_columns, imputation_values):
    """Prepara los datos de entrada para predicción"""
    input_data = {}
    for feature in model_columns:
        if feature.startswith('mission_'):
            input_data[feature] = 0
        elif feature in params:
            input_data[feature] = params[feature]
        elif params.get('mission') != 'Kepler' and feature in imputation_values:
            input_data[feature] = imputation_values[feature]
        elif params.get('mission') != 'Kepler' and feature.startswith('fp_flag_'):
            input_data[feature] = 0
        else:
            input_data[feature] = 0

    mission_column = f"mission_{params.get('mission', 'Kepler')}"
    if mission_column in input_data:
        input_data[mission_column] = 1

    return pd.DataFrame([input_data])[model_columns]

# --- Interfaz de Usuario ---
st.markdown('<h1><i class="fas fa-globe"></i> Módulo de Predicción</h1>', unsafe_allow_html=True)
st.markdown("### Introduce las características de un objeto de interés para clasificarlo")

# Selector de modelos
available_models = get_available_models()

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("Selecciona el modelo que deseas usar para realizar predicciones")
with col2:
    model_options = [m['name'] for m in available_models]
    selected_model_name = st.selectbox("Modelo", model_options, label_visibility="collapsed")

# Cargar el modelo seleccionado
selected_model_data = next((m for m in available_models if m['name'] == selected_model_name), None)

if selected_model_data and selected_model_data.get('file') != 'default':
    model, le, model_columns, imputation_values = load_custom_model(selected_model_data['model_file'])
    # Mostrar información del modelo personalizado
    with st.expander("Información del Modelo Seleccionado"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Algoritmo", selected_model_data.get('algorithm', 'N/A'))
        with col2:
            st.metric("Accuracy", f"{selected_model_data.get('accuracy', 0):.2%}")
        with col3:
            st.metric("Precision", f"{selected_model_data.get('avg_precision', 0):.2%}")
        with col4:
            st.metric("F1-Score", f"{selected_model_data.get('avg_f1', 0):.2%}")
else:
    model, le, model_columns, imputation_values = load_default_artifacts()
    # Mostrar información del modelo predeterminado
    default_metrics = get_default_model_metrics()
    with st.expander("Información del Modelo Seleccionado"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Algoritmo", default_metrics.get('algorithm', 'Random Forest'))
        with col2:
            st.metric("Accuracy", f"{default_metrics.get('accuracy', 0):.2%}")
        with col3:
            st.metric("Precision", f"{default_metrics.get('avg_precision', 0):.2%}")
        with col4:
            st.metric("F1-Score", f"{default_metrics.get('avg_f1', 0):.2%}")

st.markdown("---")

if model is None:
    st.error("Error: Archivos del modelo no encontrados. Asegúrate de que la carpeta 'artifacts' con 'exoplanet_model.joblib' y 'final_dataset.csv' existe.")
else:
    # Tabs para predicción individual y masiva
    tab1, tab2 = st.tabs(["Predicción Individual", "Predicción Masiva"])

    # ============= TAB 1: PREDICCIÓN INDIVIDUAL =============
    with tab1:
        # Botón de dado para cargar valores aleatorios
        col_header, col_button = st.columns([3, 1])
        with col_header:
            st.header("Parámetros del Objeto")
        with col_button:
            if st.button("Aleatorio", use_container_width=True, help="Cargar un registro aleatorio del dataset"):
                # Cargar dataset y seleccionar fila aleatoria
                df_sample = pd.read_csv('artifacts/final_dataset.csv')
                random_row = df_sample.sample(n=1).iloc[0]

                # Guardar valores en session_state
                st.session_state['random_mission'] = random_row['mission']
                st.session_state['random_orbital_period'] = float(random_row['orbital_period'])
                st.session_state['random_planet_radius_earth'] = float(random_row['planet_radius_earth'])
                st.session_state['random_planet_temp'] = int(random_row['planet_temp'])
                st.session_state['random_planet_count_in_system'] = int(random_row['planet_count_in_system'])
                st.session_state['random_transit_depth'] = float(random_row['transit_depth'])
                st.session_state['random_transit_duration'] = float(random_row['transit_duration'])
                st.session_state['random_impact_parameter'] = float(random_row['impact_parameter'])
                st.session_state['random_stellar_temperature'] = int(random_row['stellar_temperature'])
                st.session_state['random_stellar_radius'] = float(random_row['stellar_radius'])
                st.session_state['random_stellar_mass'] = float(random_row['stellar_mass'])
                st.session_state['random_stellar_logg'] = float(random_row['stellar_logg'])

                # Valores de Kepler si aplica
                if random_row['mission'] == 'Kepler':
                    st.session_state['random_disposition_score'] = float(random_row['disposition_score'])
                    st.session_state['random_signal_to_noise'] = float(random_row['signal_to_noise'])
                    st.session_state['random_fp_flag_nt'] = int(random_row['fp_flag_nt'])
                    st.session_state['random_fp_flag_ss'] = int(random_row['fp_flag_ss'])
                    st.session_state['random_fp_flag_co'] = int(random_row['fp_flag_co'])
                    st.session_state['random_fp_flag_ec'] = int(random_row['fp_flag_ec'])

                st.rerun()

        params = {}

        # --- Fila 1: Parámetros Globales ---
        st.markdown('<h3><i class="fas fa-check-circle"></i> Características Globales</h3>', unsafe_allow_html=True)
        st.caption("Comunes a todas las misiones espaciales")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            mission_options = ["Kepler", "K2", "TESS"]
            mission_index = mission_options.index(st.session_state.get('random_mission', 'Kepler')) if 'random_mission' in st.session_state else 0
            params['mission'] = st.selectbox("Misión de Origen", mission_options, index=mission_index, help="Misión espacial que detectó el objeto")
        with col2:
            params['orbital_period'] = st.number_input("Periodo Orbital", min_value=0.0, value=st.session_state.get('random_orbital_period', 10.5), format="%.4f", help="Tiempo que tarda en orbitar su estrella (días)")
        with col3:
            params['planet_radius_earth'] = st.number_input("Radio del Planeta", min_value=0.0, value=st.session_state.get('random_planet_radius_earth', 1.6), help="Tamaño relativo a la Tierra (Radios ⊕)")
        with col4:
            params['planet_temp'] = st.number_input("Temperatura", min_value=0, value=st.session_state.get('random_planet_temp', 1000), help="Temperatura estimada del planeta (K)")
        with col5:
            params['planet_count_in_system'] = st.number_input("Planetas en Sistema", min_value=1, value=st.session_state.get('random_planet_count_in_system', 1), step=1, help="Número de planetas detectados en el sistema")

        st.markdown("")  # Espaciado

        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            params['transit_depth'] = st.number_input("Profundidad Tránsito", min_value=0.0, value=st.session_state.get('random_transit_depth', 500.0), help="Caída de brillo durante el tránsito (ppm)")
        with col2:
            params['transit_duration'] = st.number_input("Duración Tránsito", min_value=0.0, value=st.session_state.get('random_transit_duration', 3.5), help="Duración del tránsito (horas)")
        with col3:
            params['impact_parameter'] = st.slider("Parámetro de Impacto", 0.0, 2.0, st.session_state.get('random_impact_parameter', 0.5), 0.01, help="Geometría del tránsito (0=central)")
        with col4:
            params['stellar_temperature'] = st.number_input("Temp. Estrella", min_value=2000, value=st.session_state.get('random_stellar_temperature', 5778), help="Temperatura de la estrella anfitriona (K)")
        with col5:
            params['stellar_radius'] = st.number_input("Radio Estrella", min_value=0.0, value=st.session_state.get('random_stellar_radius', 1.0), help="Tamaño relativo al Sol (Radios ☉)")
        with col6:
            params['stellar_mass'] = st.number_input("Masa Estrella", min_value=0.0, value=st.session_state.get('random_stellar_mass', 1.0), help="Masa relativa al Sol (Masas ☉)")
        with col7:
            params['stellar_logg'] = st.number_input("Gravedad Estelar", min_value=0.0, value=st.session_state.get('random_stellar_logg', 4.4), help="Gravedad superficial de la estrella (log g)")

        st.divider()

        # --- Fila 2: Características No Comunes ---
        st.markdown('<h3><i class="fas fa-satellite"></i> Características No Comunes (se habilitan según la misión)</h3>', unsafe_allow_html=True)
        if params['mission'] == 'Kepler':
            st.info("Estas características de diagnóstico solo están disponibles para la misión Kepler y mejoran significativamente la predicción.")
            k_col1, k_col2, k_col3, k_col4, k_col5, k_col6 = st.columns(6)
            with k_col1:
                params['disposition_score'] = st.slider("Score de Disposición", 0.0, 1.0, st.session_state.get('random_disposition_score', 0.95), 0.01, help="Score de disposición (0.0-1.0)")
            with k_col2:
                params['signal_to_noise'] = st.number_input("Señal-Ruido", min_value=0.0, value=st.session_state.get('random_signal_to_noise', 50.0), help="Relación señal-ruido (SNR)")
            with k_col3:
                params['fp_flag_nt'] = st.selectbox("Flag NT", [0, 1], index=st.session_state.get('random_fp_flag_nt', 0), help="Not Transit-Like Flag")
            with k_col4:
                params['fp_flag_ss'] = st.selectbox("Flag SS", [0, 1], index=st.session_state.get('random_fp_flag_ss', 0), help="Stellar Eclipse Flag")
            with k_col5:
                params['fp_flag_co'] = st.selectbox("Flag CO", [0, 1], index=st.session_state.get('random_fp_flag_co', 0), help="Centroid Offset Flag")
            with k_col6:
                params['fp_flag_ec'] = st.selectbox("Flag EC", [0, 1], index=st.session_state.get('random_fp_flag_ec', 0), help="Ephemeris Contamination Flag")
        else:
            st.warning(f"La misión '{params['mission']}' no proporciona estas características de diagnóstico. Se utilizarán valores neutros para la predicción.")

        st.divider()

        # Botón de predicción centrado
        st.markdown("")  # Espaciado
        _, center_col, _ = st.columns([1.5, 1, 1.5])
        if center_col.button("Clasificar Objeto", use_container_width=True, type="primary"):
            # Preparar datos
            input_df = prepare_input_data(params, model_columns, imputation_values)

            prediction_encoded = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            prediction_label = le.inverse_transform(prediction_encoded)[0]

            # Mostrar Resultado con diseño mejorado
            st.markdown("---")
            st.markdown('<h2><i class="fas fa-chart-pie"></i> Resultado de la Clasificación</h2>', unsafe_allow_html=True)
            st.markdown("")

            # Resultado principal con métricas grandes
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                if prediction_label == 'CONFIRMED':
                    st.success("### EXOPLANETA CONFIRMADO")
                    st.markdown('<h1 style="text-align: center;"><i class="fas fa-globe"></i></h1>', unsafe_allow_html=True)
                    confidence = prediction_proba[0][list(le.classes_).index(prediction_label)]
                    st.metric(
                        label="Nivel de Confianza",
                        value=f"{confidence:.1%}",
                        delta="Alta certeza" if confidence > 0.8 else "Certeza moderada"
                    )
                elif prediction_label == 'CANDIDATE':
                    st.info("### CANDIDATO A EXOPLANETA")
                    st.markdown('<h1 style="text-align: center;"><i class="fas fa-search"></i></h1>', unsafe_allow_html=True)
                    confidence = prediction_proba[0][list(le.classes_).index(prediction_label)]
                    st.metric(
                        label="Nivel de Confianza",
                        value=f"{confidence:.1%}",
                        delta="Requiere confirmación"
                    )
                else:
                    st.error("### FALSO POSITIVO")
                    st.markdown('<h1 style="text-align: center;"><i class="fas fa-star"></i></h1>', unsafe_allow_html=True)
                    confidence = prediction_proba[0][list(le.classes_).index(prediction_label)]
                    st.metric(
                        label="Nivel de Confianza",
                        value=f"{confidence:.1%}",
                        delta="No es un exoplaneta"
                    )

            st.markdown("---")

            # Distribución de probabilidades con gráfico
            st.markdown('<h3><i class="fas fa-chart-bar"></i> Distribución de Probabilidades</h3>', unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1])

            with col1:
                # Tabla de probabilidades
                proba_df = pd.DataFrame({
                    'Clase': le.classes_,
                    'Probabilidad': prediction_proba[0]
                }).sort_values('Probabilidad', ascending=False)

                st.dataframe(
                    proba_df.style.format({'Probabilidad': '{:.2%}'}).background_gradient(cmap='RdYlGn', subset=['Probabilidad']),
                    use_container_width=True,
                    hide_index=True
                )

            with col2:
                # Gráfico de barras con Plotly
                fig = go.Figure(data=[
                    go.Bar(
                        x=le.classes_,
                        y=prediction_proba[0],
                        text=[f'{p:.1%}' for p in prediction_proba[0]],
                        textposition='auto',
                        marker=dict(
                            color=prediction_proba[0],
                            colorscale='RdYlGn',
                            showscale=False
                        )
                    )
                ])

                fig.update_layout(
                    title="Confianza por Clase",
                    xaxis_title="Clasificación",
                    yaxis_title="Probabilidad",
                    yaxis=dict(tickformat='.0%'),
                    height=300,
                    showlegend=False,
                    template="plotly_dark"
                )

                st.plotly_chart(fig, use_container_width=True)

            # XAI: Feature Importance para esta predicción
            st.markdown("---")
            st.markdown('<h3><i class="fas fa-brain"></i> Explicabilidad: ¿Por qué esta predicción?</h3>', unsafe_allow_html=True)

            if hasattr(model, 'feature_importances_'):
                st.markdown("**Contribución de Features a la Predicción**")

                # Mostrar top features importantes y sus valores
                feature_values = input_df.iloc[0]
                feature_imp_global = pd.DataFrame({
                    'Feature': model_columns,
                    'Importancia Global': model.feature_importances_,
                    'Valor en Predicción': feature_values.values
                }).sort_values('Importancia Global', ascending=False).head(15)

                col1, col2 = st.columns(2)

                with col1:
                    fig_feat = go.Figure(go.Bar(
                        y=feature_imp_global['Feature'],
                        x=feature_imp_global['Importancia Global'],
                        orientation='h',
                        marker=dict(color='lightblue'),
                        text=[f'{v:.3f}' for v in feature_imp_global['Importancia Global']],
                        textposition='auto'
                    ))

                    fig_feat.update_layout(
                        title="Top 15 Features Más Influyentes (Global)",
                        xaxis_title="Importancia",
                        yaxis_title="Feature",
                        height=450,
                        template="plotly_dark",
                        yaxis={'categoryorder': 'total ascending'}
                    )

                    st.plotly_chart(fig_feat, use_container_width=True)

                with col2:
                    st.markdown("**Valores de los Features Más Importantes**")
                    display_df = feature_imp_global[['Feature', 'Valor en Predicción', 'Importancia Global']].copy()
                    display_df['Importancia Global'] = display_df['Importancia Global'].apply(lambda x: f'{x:.4f}')
                    display_df['Valor en Predicción'] = display_df['Valor en Predicción'].apply(lambda x: f'{x:.4f}')

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        height=450
                    )

            # Interpretación del resultado
            st.markdown("---")
            with st.expander("¿Cómo interpretar estos resultados?"):
                st.markdown("""
                **CONFIRMED (Confirmado):** El objeto ha sido verificado como un exoplaneta real con alta confianza.

                **CANDIDATE (Candidato):** Muestra características prometedoras pero requiere observaciones adicionales para confirmación.

                **FALSE POSITIVE (Falso Positivo):** El objeto no es un exoplaneta, probablemente una estrella binaria eclipsante u otro fenómeno.

                **Nivel de Confianza:** Indica qué tan seguro está el modelo de su predicción. Valores >80% indican alta certeza.

                **Feature Importance:** Muestra qué características del planeta tienen mayor peso en la decisión del modelo.
                Las features con mayor importancia global son las que más influyen en las predicciones del modelo en general.
                """)

    # ============= TAB 2: PREDICCIÓN MASIVA =============
    with tab2:
        st.markdown('<h2><i class="fas fa-chart-pie"></i> Predicción Masiva de Exoplanetas</h2>', unsafe_allow_html=True)
        st.markdown("Sube un archivo CSV con múltiples candidatos para clasificarlos en lote")

        # Upload CSV
        uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=['csv'], help="El archivo debe contener las mismas columnas que el dataset de entrenamiento")

        if uploaded_file is not None:
            try:
                # Leer CSV
                df_batch = pd.read_csv(uploaded_file)

                st.success(f"Archivo cargado: {len(df_batch)} registros detectados")

                # Mostrar preview
                with st.expander("Vista Previa de los Datos"):
                    st.dataframe(df_batch.head(10), use_container_width=True)

                # Validar columnas necesarias
                required_base_cols = ['orbital_period', 'planet_radius_earth', 'planet_temp', 'stellar_temperature', 'stellar_radius', 'stellar_mass']
                missing_cols = [col for col in required_base_cols if col not in df_batch.columns]

                if missing_cols:
                    st.error(f"Faltan columnas requeridas: {', '.join(missing_cols)}")
                else:
                    if st.button("Realizar Predicciones Masivas", type="primary", use_container_width=True):
                        with st.spinner("Procesando predicciones... Esto puede tardar un momento."):
                            # Preparar datos para predicción
                            df_processed = df_batch.copy()

                            # Crear dummies para mission si existe
                            if 'mission' in df_processed.columns:
                                df_processed = pd.get_dummies(df_processed, columns=['mission'])

                            # Asegurarse de que todas las columnas del modelo están presentes
                            for col in model_columns:
                                if col not in df_processed.columns:
                                    if col.startswith('mission_'):
                                        df_processed[col] = 0
                                    elif col in imputation_values:
                                        df_processed[col] = imputation_values[col]
                                    elif col.startswith('fp_flag_'):
                                        df_processed[col] = 0
                                    else:
                                        df_processed[col] = 0

                            # Reordenar columnas
                            X_batch = df_processed[model_columns]

                            # Hacer predicciones
                            predictions = model.predict(X_batch)
                            predictions_proba = model.predict_proba(X_batch)
                            prediction_labels = le.inverse_transform(predictions)

                            # Agregar resultados al dataframe original
                            results_df = df_batch.copy()
                            results_df['Predicción'] = prediction_labels
                            results_df['Confianza'] = [predictions_proba[i][predictions[i]] for i in range(len(predictions))]

                            # Agregar probabilidades por clase
                            for i, class_name in enumerate(le.classes_):
                                results_df[f'Prob_{class_name}'] = predictions_proba[:, i]

                        st.success(f"¡Predicciones completadas! {len(results_df)} objetos clasificados")

                        # Dashboard de Resultados
                        st.markdown("---")
                        st.markdown('<h2><i class="fas fa-tachometer-alt"></i> Dashboard de Resultados</h2>', unsafe_allow_html=True)

                        # Métricas generales
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            confirmed_pct = (prediction_labels == 'CONFIRMED').sum() / len(prediction_labels)
                            st.metric(
                                label="% CONFIRMED",
                                value=f"{confirmed_pct:.1%}",
                                delta=f"{(prediction_labels == 'CONFIRMED').sum()} objetos"
                            )

                        with col2:
                            candidate_pct = (prediction_labels == 'CANDIDATE').sum() / len(prediction_labels)
                            st.metric(
                                label="% CANDIDATE",
                                value=f"{candidate_pct:.1%}",
                                delta=f"{(prediction_labels == 'CANDIDATE').sum()} objetos"
                            )

                        with col3:
                            fp_pct = (prediction_labels == 'FALSE POSITIVE').sum() / len(prediction_labels)
                            st.metric(
                                label="% FALSE POSITIVE",
                                value=f"{fp_pct:.1%}",
                                delta=f"{(prediction_labels == 'FALSE POSITIVE').sum()} objetos"
                            )

                        with col4:
                            avg_confidence = results_df['Confianza'].mean()
                            st.metric(
                                label="Confianza Promedio",
                                value=f"{avg_confidence:.1%}",
                                delta="Media general"
                            )

                        st.markdown("---")

                        # Visualizaciones
                        col1, col2 = st.columns(2)

                        with col1:
                            # Distribución de predicciones
                            pred_counts = pd.Series(prediction_labels).value_counts()

                            fig_pie = px.pie(
                                values=pred_counts.values,
                                names=pred_counts.index,
                                title="Distribución de Clasificaciones",
                                hole=0.4,
                                template="plotly_dark",
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            fig_pie.update_traces(textposition='inside', textinfo='percent+label+value')

                            st.plotly_chart(fig_pie, use_container_width=True)

                        with col2:
                            # Histograma de confianza
                            fig_hist = go.Figure()

                            for class_name in le.classes_:
                                class_confidences = results_df[results_df['Predicción'] == class_name]['Confianza']
                                fig_hist.add_trace(go.Histogram(
                                    x=class_confidences,
                                    name=class_name,
                                    opacity=0.7,
                                    nbinsx=20
                                ))

                            fig_hist.update_layout(
                                title="Distribución de Niveles de Confianza por Clase",
                                xaxis_title="Confianza",
                                yaxis_title="Frecuencia",
                                barmode='overlay',
                                template="plotly_dark",
                                xaxis=dict(tickformat='.0%')
                            )

                            st.plotly_chart(fig_hist, use_container_width=True)

                        # Box plot de probabilidades
                        st.markdown('<h3><i class="fas fa-box"></i> Distribución de Probabilidades por Clase Predicha</h3>', unsafe_allow_html=True)

                        prob_data = []
                        for pred in le.classes_:
                            for actual_class in le.classes_:
                                class_probs = results_df[results_df['Predicción'] == pred][f'Prob_{actual_class}']
                                for prob in class_probs:
                                    prob_data.append({
                                        'Predicción': pred,
                                        'Clase': actual_class,
                                        'Probabilidad': prob
                                    })

                        prob_df = pd.DataFrame(prob_data)

                        fig_box = px.box(
                            prob_df,
                            x='Predicción',
                            y='Probabilidad',
                            color='Clase',
                            title="Distribución de Probabilidades",
                            template="plotly_dark"
                        )

                        fig_box.update_layout(yaxis=dict(tickformat='.0%'), height=400)

                        st.plotly_chart(fig_box, use_container_width=True)

                        st.markdown("---")

                        # Tabla de resultados interactiva
                        st.markdown('<h3><i class="fas fa-table"></i> Tabla de Resultados Completa</h3>', unsafe_allow_html=True)

                        # Filtros
                        col1, col2 = st.columns(2)

                        with col1:
                            filter_class = st.multiselect(
                                "Filtrar por Clasificación",
                                options=list(le.classes_),
                                default=list(le.classes_)
                            )

                        with col2:
                            min_confidence = st.slider(
                                "Confianza Mínima",
                                0.0, 1.0, 0.0, 0.05,
                                help="Filtrar resultados por nivel mínimo de confianza"
                            )

                        # Aplicar filtros
                        filtered_results = results_df[
                            (results_df['Predicción'].isin(filter_class)) &
                            (results_df['Confianza'] >= min_confidence)
                        ]

                        st.markdown(f"**Mostrando {len(filtered_results)} de {len(results_df)} registros**")

                        # Formatear y mostrar tabla
                        display_results = filtered_results.copy()
                        display_results['Confianza'] = display_results['Confianza'].apply(lambda x: f'{x:.2%}')

                        for class_name in le.classes_:
                            if f'Prob_{class_name}' in display_results.columns:
                                display_results[f'Prob_{class_name}'] = display_results[f'Prob_{class_name}'].apply(lambda x: f'{x:.2%}')

                        st.dataframe(
                            display_results,
                            use_container_width=True,
                            height=400
                        )

                        # Opción de descarga
                        st.markdown("---")
                        st.markdown('<h3><i class="fas fa-download"></i> Exportar Resultados</h3>', unsafe_allow_html=True)

                        csv = results_df.to_csv(index=False).encode('utf-8')

                        st.download_button(
                            label="Descargar Resultados en CSV",
                            data=csv,
                            file_name=f'predicciones_exoplanetas_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv',
                            mime='text/csv',
                            use_container_width=True
                        )

                        # Si hay columna 'disposition' real en el CSV, hacer comparación
                        if 'disposition' in df_batch.columns:
                            st.markdown("---")
                            st.markdown('<h3><i class="fas fa-crosshairs"></i> Comparación con Ground Truth</h3>', unsafe_allow_html=True)
                            st.markdown("Se detectó una columna 'disposition' en tu CSV. Comparando predicciones con valores reales...")

                            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

                            y_true = df_batch['disposition'].values
                            y_pred = prediction_labels

                            accuracy = accuracy_score(y_true, y_pred)
                            cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
                            report = classification_report(y_true, y_pred, output_dict=True)

                            # Métricas
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Accuracy", f"{accuracy:.2%}")
                            with col2:
                                st.metric("Precision (promedio)", f"{report['weighted avg']['precision']:.2%}")
                            with col3:
                                st.metric("Recall (promedio)", f"{report['weighted avg']['recall']:.2%}")

                            # Matriz de confusión
                            fig_cm = go.Figure(data=go.Heatmap(
                                z=cm,
                                x=le.classes_,
                                y=le.classes_,
                                colorscale='RdYlGn',
                                text=cm,
                                texttemplate='%{text}',
                                textfont={"size": 16},
                                showscale=True
                            ))

                            fig_cm.update_layout(
                                title="Matriz de Confusión: Real vs Predicho",
                                xaxis_title="Predicción",
                                yaxis_title="Valor Real",
                                height=400,
                                template="plotly_dark"
                            )

                            st.plotly_chart(fig_cm, use_container_width=True)

            except Exception as e:
                st.error(f"Error al procesar el archivo: {str(e)}")
                st.info("Asegúrate de que el CSV tiene el formato correcto y las columnas necesarias")
