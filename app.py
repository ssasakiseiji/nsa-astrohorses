# app.py
import streamlit as st

st.set_page_config(
    page_title="Exoplanet AI Workbench",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar Font Awesome
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

# Header principal con estilo mejorado
st.markdown('<h1><i class="fas fa-satellite"></i> Exoplanet AI Workbench</h1>', unsafe_allow_html=True)
st.markdown("### Una Plataforma Interactiva para la Detección de Exoplanetas")

st.markdown("---")

# Sección de bienvenida
st.markdown("""
Bienvenido al **Workbench de IA para Exoplanetas**. Esta plataforma te permite utilizar modelos de Machine Learning y configurar otros con hiperparámetros personalizados para ayudar en la búsqueda de nuevos mundos.
""")

st.markdown("")  # Espaciado

# Métricas destacadas del proyecto
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<i class="fas fa-bullseye"></i> **Precisión del Modelo**', unsafe_allow_html=True)
    st.metric(
        label="Precisión del Modelo",
        value="89,7%",
        delta="Pre-entrenado",
        help="Modelo Random Forest optimizado",
        label_visibility="collapsed"
    )
with col2:
    st.markdown('<i class="fas fa-satellite"></i> **Misiones Soportadas**', unsafe_allow_html=True)
    st.metric(
        label="Misiones Soportadas",
        value="3",
        delta="Kepler, K2, TESS",
        help="Tres misiones de telescopios espaciales",
        label_visibility="collapsed"
    )
with col3:
    st.markdown('<i class="fas fa-cogs"></i> **Algoritmos Disponibles**', unsafe_allow_html=True)
    st.metric(
        label="Algoritmos Disponibles",
        value="2",
        delta="Random Forest+F & LightGBM",
        help="Random Forest y LightGBM para entrenamiento",
        label_visibility="collapsed"
    )

st.markdown("---")

# Módulos disponibles con mejor presentación
st.markdown('<h3><i class="fas fa-th-large"></i> Módulos Disponibles</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown('<h4><i class="fas fa-globe"></i> Predictor</h4>', unsafe_allow_html=True)
        st.markdown("""
        Utiliza nuestro modelo pre-entrenado con **89,7% de precisión** para clasificar objetos de interés.

        **Características:**
        - Clasificación en tiempo real
        - Soporte para 3 misiones espaciales
        - Parámetros personalizables
        - Análisis de predicción detallado
        """)
        st.info("Selecciona 'Predictor' en la barra lateral para comenzar")

with col2:
    with st.container():
        st.markdown('<h4><i class="fas fa-cog"></i> Model Workshop</h4>', unsafe_allow_html=True)
        st.markdown("""
        Entrena y evalúa tus propios modelos de clasificación.

        **Características:**
        - Ajuste de hiperparámetros en tiempo real
        - Algoritmos RandomForest y LightGBM
        - Métricas de rendimiento detalladas
        - Visualizaciones interactivas
        """)
        st.info("Selecciona 'Model Workshop' en la barra lateral")

st.markdown("---")

# Footer con información adicional
st.markdown('<details><summary><i class="fas fa-info-circle"></i> <b>Sobre este Proyecto</b></summary>', unsafe_allow_html=True)
with st.expander("Sobre este Proyecto", expanded=False):
    st.markdown("""
    **Exoplanet AI Workbench** es una plataforma educativa e interactiva para explorar técnicas de Machine Learning
    aplicadas a la astronomía y la búsqueda de exoplanetas.

    Los datos provienen de misiones espaciales reales (Kepler, K2, TESS) y los modelos están entrenados
    para identificar patrones que indican la presencia de planetas extrasolares.
    """)