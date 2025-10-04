# pages/2_⚙️_Model_Workshop.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

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
st.markdown("### Entrena y evalúa tus propios modelos de clasificación")
st.markdown("Ajusta los hiperparámetros en la barra lateral, entrena y evalúa el rendimiento con visualizaciones en tiempo real.")
st.markdown("---")

if df is None:
    st.error("Dataset no encontrado. Asegúrate de que 'final_dataset.csv' está en la carpeta 'artifacts'.")
else:
    # Mostrar información del dataset
    st.subheader("📊 Información del Dataset")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Muestras", f"{len(df):,}")
    with col2:
        st.metric("Características", len(df.columns) - 1)
    with col3:
        st.metric("Clases", df['disposition'].nunique())
    with col4:
        st.metric("Misiones", df['mission'].nunique())

    # Distribución de clases
    with st.expander("📈 Ver Distribución de Clases"):
        class_counts = df['disposition'].value_counts()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(
                pd.DataFrame({
                    'Clase': class_counts.index,
                    'Cantidad': class_counts.values,
                    'Porcentaje': (class_counts.values / len(df) * 100).round(2)
                }),
                use_container_width=True,
                hide_index=True
            )

        with col2:
            fig = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title="Distribución de Clases en el Dataset",
                hole=0.4,
                template="plotly_dark"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.sidebar.header("⚙️ Configuración del Entrenamiento")

    model_name = st.sidebar.text_input("Nombre de tu Modelo", "Mi Modelo Personalizado", help="Dale un nombre a tu modelo")
    algorithm = st.sidebar.selectbox("Algoritmo", ["RandomForest", "LightGBM"], help="Selecciona el algoritmo de ML")

    st.sidebar.subheader("🎛️ Ajuste de Hiperparámetros")
    params = {}
    if algorithm == "RandomForest":
        params['n_estimators'] = st.sidebar.slider("Número de Árboles", 50, 500, 100, 10, help="Más árboles = mejor precisión pero más lento")
        params['max_depth'] = st.sidebar.slider("Profundidad Máxima", 5, 50, 10, 1, help="Profundidad de cada árbol")
    elif algorithm == "LightGBM":
        params['n_estimators'] = st.sidebar.slider("Número de Estimadores", 50, 500, 100, 10, help="Número de boosting rounds")
        params['learning_rate'] = st.sidebar.slider("Tasa de Aprendizaje", 0.01, 0.3, 0.1, 0.01, help="Qué tan rápido aprende el modelo")
        params['num_leaves'] = st.sidebar.slider("Número de Hojas", 20, 100, 31, 1, help="Complejidad del árbol")

    if st.sidebar.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
        st.header(f"📊 Resultados para: {model_name}")
        st.markdown("---")

        with st.spinner("🔄 Preparando datos y entrenando el modelo... Esto puede tardar unos segundos."):
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
            y_pred_proba = model.predict_proba(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred), output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

        st.success(f"✅ ¡Entrenamiento completado con éxito!")

        # Métricas principales destacadas
        st.subheader("🎯 Métricas Principales")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Precisión General",
                value=f"{accuracy:.2%}",
                delta="Accuracy",
                help="Porcentaje de predicciones correctas"
            )

        # Extraer métricas promedio
        avg_precision = report['weighted avg']['precision']
        avg_recall = report['weighted avg']['recall']
        avg_f1 = report['weighted avg']['f1-score']

        with col2:
            st.metric(
                label="Precision (Promedio)",
                value=f"{avg_precision:.2%}",
                help="Precisión promedio ponderada"
            )

        with col3:
            st.metric(
                label="Recall (Promedio)",
                value=f"{avg_recall:.2%}",
                help="Recall promedio ponderado"
            )

        with col4:
            st.metric(
                label="F1-Score (Promedio)",
                value=f"{avg_f1:.2%}",
                help="Media armónica de precision y recall"
            )

        st.markdown("---")

        # Gráficos y visualizaciones
        col1, col2 = st.columns(2)

        with col1:
            # Matriz de confusión
            st.subheader("📊 Matriz de Confusión")

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
                xaxis_title="Predicción",
                yaxis_title="Valor Real",
                height=400,
                template="plotly_dark"
            )

            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            # Métricas por clase
            st.subheader("📈 Métricas por Clase")

            # Extraer métricas de cada clase
            classes = le.classes_
            metrics_data = []

            for cls in classes:
                if cls in report:
                    metrics_data.append({
                        'Clase': cls,
                        'Precision': report[cls]['precision'],
                        'Recall': report[cls]['recall'],
                        'F1-Score': report[cls]['f1-score']
                    })

            metrics_df = pd.DataFrame(metrics_data)

            # Gráfico de barras agrupadas
            fig_metrics = go.Figure()

            fig_metrics.add_trace(go.Bar(
                name='Precision',
                x=metrics_df['Clase'],
                y=metrics_df['Precision'],
                marker_color='rgb(99, 110, 250)'
            ))

            fig_metrics.add_trace(go.Bar(
                name='Recall',
                x=metrics_df['Clase'],
                y=metrics_df['Recall'],
                marker_color='rgb(239, 85, 59)'
            ))

            fig_metrics.add_trace(go.Bar(
                name='F1-Score',
                x=metrics_df['Clase'],
                y=metrics_df['F1-Score'],
                marker_color='rgb(0, 204, 150)'
            ))

            fig_metrics.update_layout(
                barmode='group',
                yaxis_title='Score',
                xaxis_title='Clase',
                height=400,
                yaxis=dict(tickformat='.0%'),
                template="plotly_dark"
            )

            st.plotly_chart(fig_metrics, use_container_width=True)

        st.markdown("---")

        # Tabla detallada del reporte
        st.subheader("📋 Reporte de Clasificación Detallado")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(
            report_df.style.format("{:.3f}").background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']),
            use_container_width=True
        )

        # Interpretación
        st.markdown("---")
        with st.expander("💡 ¿Cómo interpretar estas métricas?"):
            st.markdown(f"""
            ### Métricas de Rendimiento

            **Accuracy (Precisión General):** {accuracy:.2%}
            - Porcentaje total de predicciones correctas sobre el total de muestras.

            **Precision:** Cuando el modelo predice una clase, ¿qué tan seguido acierta?
            - Precision alta = Pocas falsas alarmas.

            **Recall (Sensibilidad):** De todos los casos reales de una clase, ¿cuántos identifica el modelo?
            - Recall alto = Pocas detecciones perdidas.

            **F1-Score:** Media armónica entre precision y recall.
            - Equilibra ambas métricas para una evaluación completa.

            **Matriz de Confusión:** Muestra dónde el modelo acierta y se confunde.
            - Diagonal = Predicciones correctas
            - Fuera de la diagonal = Errores de clasificación

            ### Configuración de tu Modelo

            **Algoritmo:** {algorithm}
            **Parámetros:** {params}
            """)