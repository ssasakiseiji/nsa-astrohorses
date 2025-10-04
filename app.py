# app.py
import streamlit as st

st.set_page_config(
    page_title="Exoplanet AI Workbench",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Exoplanet AI Workbench")
st.header("Una Plataforma Interactiva para la Detección de Exoplanetas")

st.markdown("""
Bienvenido al Workbench de IA para Exoplanetas. Esta plataforma te permite utilizar modelos de Machine Learning de última generación y construir los tuyos propios para ayudar en la búsqueda de nuevos mundos.

**Navega a los diferentes módulos utilizando la barra lateral a la izquierda:**

- **🪐 Predictor:** Utiliza nuestro modelo pre-entrenado (~90% de precisión) para clasificar un objeto de interés introduciendo sus características.
- **⚙️ Model Workshop:** ¡Conviértete en el científico de datos! Entrena y evalúa tus propios modelos de clasificación, personalizando los algoritcopmos y sus hiperparámetros en tiempo real.
""")

st.info("Para comenzar, selecciona un módulo del menú de navegación en la barra lateral.", icon="👈")