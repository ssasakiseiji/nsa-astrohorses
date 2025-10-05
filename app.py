# app.py
import streamlit as st
from sidebar_config import setup_sidebar

st.set_page_config(
    page_title="PlanetHunter",
    page_icon=":material/public:",  # Icono de planeta/globo de Material Symbols
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar sidebar consistente
setup_sidebar()

# Header principal con estilo mejorado
st.markdown('<h1><i class="fas fa-satellite"></i> PlanetHunter</h1>', unsafe_allow_html=True)
st.markdown("### Plataforma de Inteligencia Artificial para la Detección de Exoplanetas")

st.markdown("---")

# Accesos directos a las páginas principales
st.markdown('<h3><i class="fas fa-compass"></i> Acceso Rápido</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <a href="/Predictor" target="_self" style="text-decoration: none;">
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            cursor: pointer;
            transition: transform 0.2s;
        ">
            <i class="fas fa-search" style="font-size: 3rem; margin-bottom: 1rem;"></i>
            <h3 style="margin: 0.5rem 0;">Predictor</h3>
            <p style="margin: 0; opacity: 0.9;">Clasifica objetos astronómicos</p>
        </div>
    </a>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <a href="/Model_Workshop" target="_self" style="text-decoration: none;">
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            cursor: pointer;
            transition: transform 0.2s;
        ">
            <i class="fas fa-tools" style="font-size: 3rem; margin-bottom: 1rem;"></i>
            <h3 style="margin: 0.5rem 0;">Model Workshop</h3>
            <p style="margin: 0; opacity: 0.9;">Entrena tus propios modelos</p>
        </div>
    </a>
    """, unsafe_allow_html=True)

st.markdown("---")

# Overview del Proyecto
st.markdown('<h3><i class="fas fa-chart-line"></i> Overview del Proyecto</h3>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Precisión del Modelo",
        value="89.7%",
        delta="Accuracy",
        help="Modelo Random Forest optimizado"
    )
with col2:
    st.metric(
        label="Muestras Entrenadas",
        value="9,564",
        delta="Objetos",
        help="Total de candidatos en el dataset"
    )
with col3:
    st.metric(
        label="Misiones Espaciales",
        value="3",
        delta="Kepler, K2, TESS",
        help="Telescopios espaciales soportados"
    )
with col4:
    st.metric(
        label="Algoritmos ML",
        value="2",
        delta="RF & LightGBM",
        help="Random Forest y LightGBM disponibles"
    )

st.markdown("---")

# Explicación del Proyecto
st.markdown('<h3><i class="fas fa-info-circle"></i> Sobre PlanetHunter</h3>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["¿Qué es PlanetHunter?", "¿Cómo funciona?", "Módulos Disponibles"])

with tab1:
    st.markdown("""
    ### La Búsqueda de Nuevos Mundos

    **PlanetHunter** es una plataforma de inteligencia artificial diseñada para ayudar en la detección y clasificación
    de exoplanetas utilizando datos reales de misiones espaciales.

    #### ¿Qué son los Exoplanetas?
    Los exoplanetas son planetas que orbitan estrellas fuera de nuestro Sistema Solar. Hasta la fecha, se han confirmado
    más de 5,000 exoplanetas, y cada nuevo descubrimiento nos acerca más a responder una de las preguntas más antiguas
    de la humanidad: **¿Estamos solos en el universo?**

    #### Nuestra Misión
    Proveer el acceso a las herramientas de Machine Learning para la astronomía, permitiendo a investigadores,
    estudiantes y entusiastas explorar y contribuir al fascinante campo de la detección de exoplanetas.

    #### Datos Reales
    Trabajamos con datos de tres misiones espaciales de la NASA:
    - **Kepler**: El pionero en la búsqueda de exoplanetas (2009-2018)
    - **K2**: La extensión de la misión Kepler (2014-2018)
    - **TESS**: Transiting Exoplanet Survey Satellite (2018-presente)
    """)

with tab2:
    st.markdown("""
    ### El Proceso de Detección por IA

    #### 1. Recolección de Datos
    Los telescopios espaciales monitorean miles de estrellas, midiendo su brillo de forma continua. Cuando un planeta
    cruza frente a su estrella (tránsito), causa una pequeña disminución en el brillo.

    #### 2. Características Analizadas
    Nuestro modelo analiza múltiples parámetros:

    **Características del Tránsito:**
    - Profundidad del tránsito (caída de brillo)
    - Duración del tránsito
    - Periodo orbital
    - Parámetro de impacto

    **Propiedades del Candidato:**
    - Radio del planeta
    - Temperatura estimada
    - Número de planetas en el sistema

    **Propiedades Estelares:**
    - Temperatura de la estrella
    - Radio y masa estelar
    - Gravedad superficial (log g)

    **Indicadores de Calidad (Kepler):**
    - Score de disposición
    - Relación señal-ruido
    - Flags de falsos positivos

    #### 3. Clasificación por Machine Learning
    El modelo Random Forest entrenado clasifica cada objeto en tres categorías:
    - **CONFIRMED**: Exoplaneta confirmado
    - **CANDIDATE**: Candidato prometedor que requiere más observaciones
    - **FALSE POSITIVE**: No es un exoplaneta (ej: estrella binaria eclipsante)

    #### 4. Métricas de Rendimiento
    - **Accuracy**: 89.7% - Predicciones correctas del total
    - **Precision**: 90.1% - Cuando predice un exoplaneta, acierta el 90%
    - **F1-Score**: 89.7% - Balance entre precisión y exhaustividad
    """)

    st.success("**Ventaja del ML**: Procesar miles de candidatos en segundos, algo que tomaría años de análisis manual.")

with tab3:
    st.markdown("""
    ### Herramientas de PlanetHunter
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### <i class="fas fa-search"></i> Predictor

        **Funcionalidad Principal:**
        Clasifica objetos astronómicos utilizando el modelo pre-entrenado o tus modelos personalizados.

        **Características:**
        - Predicción individual con parámetros personalizables
        - Predicción masiva desde archivos CSV
        - Análisis de probabilidades por clase
        - Explicabilidad con Feature Importance
        - Función de carga aleatoria para testing
        - Exportación de resultados

        **Ideal para:**
        - Clasificar nuevos candidatos a exoplanetas
        - Validar objetos de interés
        - Análisis exploratorio de datos
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        #### <i class="fas fa-tools"></i> Model Workshop

        **Funcionalidad Principal:**
        Entrena y evalúa tus propios modelos de clasificación con hiperparámetros personalizados.

        **Características:**
        - Algoritmos: Random Forest y LightGBM
        - Ajuste de hiperparámetros en tiempo real
        - Métricas detalladas (Accuracy, Precision, Recall, F1)
        - Matriz de confusión interactiva
        - Curvas ROC y Precision-Recall
        - Análisis de Importancia de las Características
        - Detección y análisis de errores
        - Guardado de modelos para uso en Predictor

        **Ideal para:**
        - Experimentar con diferentes configuraciones
        - Optimizar rendimiento del modelo
        - Aprender sobre Machine Learning aplicado
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### Flujo de Trabajo Recomendado

    1. **Explora** el modelo predeterminado en **Predictor**
    2. **Experimenta** con valores aleatorios usando el botón de dado
    3. **Entrena** tu propio modelo en **Model Workshop**
    4. **Guarda** tu modelo personalizado
    5. **Usa** tu modelo en **Predictor** para clasificaciones
    6. **Compara** resultados entre diferentes modelos
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #666;">
    <p><i class="fas fa-satellite"></i> PlanetHunter - AI-Powered Exoplanet Detection Platform</p>
    <p style="font-size: 0.9rem;">Datos de NASA Exoplanet Archive | Misiones: Kepler, K2, TESS</p>
</div>
""", unsafe_allow_html=True)