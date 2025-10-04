# pages/2_⚙️_Model_Workshop.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Model Workshop", page_icon="⚙️", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('artifacts/final_dataset.csv')
        return df
    except FileNotFoundError:
        return None

df = load_data()

st.title("⚙️ Taller de Modelos (Model Workshop)")
st.write("Aquí puedes entrenar tus propios modelos de clasificación. Ajusta los hiperparámetros en la barra lateral, entrena y evalúa el rendimiento en tiempo real.")

if df is None:
    st.error("Dataset no encontrado. Asegúrate de que 'final_dataset.csv' está en la carpeta 'artifacts'.")
else:
    st.sidebar.header("Configuración del Entrenamiento")
    
    model_name = st.sidebar.text_input("Nombre de tu Modelo", "Mi Modelo Personalizado")
    algorithm = st.sidebar.selectbox("Algoritmo", ["RandomForest", "LightGBM"])
    
    st.sidebar.subheader("Ajuste de Hiperparámetros")
    params = {}
    if algorithm == "RandomForest":
        params['n_estimators'] = st.sidebar.slider("Número de Árboles", 50, 500, 100, 10)
        params['max_depth'] = st.sidebar.slider("Profundidad Máxima", 5, 50, 10, 1)
    elif algorithm == "LightGBM":
        params['n_estimators'] = st.sidebar.slider("Número de Estimadores", 50, 500, 100, 10)
        params['learning_rate'] = st.sidebar.slider("Tasa de Aprendizaje", 0.01, 0.3, 0.1, 0.01)
        params['num_leaves'] = st.sidebar.slider("Número de Hojas", 20, 100, 31, 1)

    if st.sidebar.button("Entrenar Modelo", type="primary"):
        st.header(f"Resultados para: {model_name}")
        
        with st.spinner("Preparando datos y entrenando el modelo... Esto puede tardar unos segundos."):
            model_df = df.copy()
            le = LabelEncoder().fit(model_df['disposition'])
            model_df['disposition'] = le.transform(model_df['disposition'])
            model_df = pd.get_dummies(model_df, columns=['mission'])

            X = model_df.drop('disposition', axis=1)
            y = model_df['disposition']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            if algorithm == "RandomForest":
                model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
            else:
                model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, **params)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred), output_dict=True)

        st.success(f"¡Entrenamiento completado! Precisión General: **{accuracy:.2%}**")
        st.subheader("Reporte de Clasificación")
        st.dataframe(pd.DataFrame(report).transpose())