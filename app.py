# app.py
import streamlit as st

st.set_page_config(
    page_title="Exoplanet AI Workbench",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header principal con estilo mejorado
st.title("🛰️ Exoplanet AI Workbench")
st.markdown("### Una Plataforma Interactiva para la Detección de Exoplanetas")

st.markdown("---")

# Sección de bienvenida
st.markdown("""
Bienvenido al **Workbench de IA para Exoplanetas**. Esta plataforma te permite utilizar modelos de Machine Learning
de última generación y construir los tuyos propios para ayudar en la búsqueda de nuevos mundos.
""")

st.markdown("")  # Espaciado

# Métricas destacadas del proyecto
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="🎯 Precisión del Modelo",
        value="~90%",
        delta="Pre-entrenado",
        help="Modelo Random Forest optimizado"
    )
with col2:
    st.metric(
        label="🛰️ Misiones Soportadas",
        value="3",
        delta="Kepler, K2, TESS",
        help="Tres misiones espaciales principales"
    )
with col3:
    st.metric(
        label="⚙️ Algoritmos Disponibles",
        value="2+",
        delta="RF & LightGBM",
        help="Random Forest y LightGBM para entrenamiento"
    )

st.markdown("---")

# Módulos disponibles con mejor presentación
st.markdown("### 📋 Módulos Disponibles")

col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown("#### 🪐 Predictor")
        st.markdown("""
        Utiliza nuestro modelo pre-entrenado con **~90% de precisión** para clasificar objetos de interés.

        **Características:**
        - Clasificación en tiempo real
        - Soporte para 3 misiones espaciales
        - Parámetros personalizables
        - Análisis de confianza detallado
        """)
        st.info("👈 Selecciona 'Predictor' en la barra lateral para comenzar", icon="🪐")

with col2:
    with st.container():
        st.markdown("#### ⚙️ Model Workshop")
        st.markdown("""
        ¡Conviértete en científico de datos! Entrena y evalúa tus propios modelos de clasificación.

        **Características:**
        - Ajuste de hiperparámetros en tiempo real
        - Algoritmos RandomForest y LightGBM
        - Métricas de rendimiento detalladas
        - Visualizaciones interactivas
        """)
        st.info("👈 Selecciona 'Model Workshop' en la barra lateral", icon="⚙️")

st.markdown("---")

# Footer con información adicional
with st.expander("ℹ️ Sobre este Proyecto"):
    st.markdown("""
    **Exoplanet AI Workbench** es una plataforma educativa e interactiva para explorar técnicas de Machine Learning
    aplicadas a la astronomía y la búsqueda de exoplanetas.

    Los datos provienen de misiones espaciales reales (Kepler, K2, TESS) y los modelos están entrenados
    para identificar patrones que indican la presencia de planetas extrasolares.
    """)