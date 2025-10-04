# pages/2_⚙️_Model_Workshop.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import joblib
import json
import os
from datetime import datetime

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

    # Configuración del Entrenamiento en el componente principal
    st.subheader("⚙️ Configuración del Entrenamiento")

    col1, col2 = st.columns(2)
    with col1:
        model_name = st.text_input("Nombre de tu Modelo", "Mi Modelo Personalizado", help="Dale un nombre a tu modelo")
    with col2:
        algorithm = st.selectbox("Algoritmo", ["RandomForest", "LightGBM"], help="Selecciona el algoritmo de ML")

    st.markdown("#### 🎛️ Ajuste de Hiperparámetros")
    params = {}
    if algorithm == "RandomForest":
        st.markdown("**Parámetros de Estructura del Árbol**")
        col1, col2, col3 = st.columns(3)
        with col1:
            params['n_estimators'] = st.slider("Número de Árboles", 50, 500, 100, 10, help="Más árboles = mejor precisión pero más lento")
        with col2:
            params['max_depth'] = st.slider("Profundidad Máxima", 5, 50, 10, 1, help="Profundidad de cada árbol")
        with col3:
            params['min_samples_split'] = st.slider("Min Muestras Split", 2, 20, 2, 1, help="Mínimo de muestras para dividir un nodo")

        st.markdown("**Parámetros de Regularización y Features**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            params['min_samples_leaf'] = st.slider("Min Muestras Hoja", 1, 10, 1, 1, help="Mínimo de muestras en hojas")
        with col2:
            max_features = st.selectbox("Max Features", ["sqrt", "log2", "None"], help="Número de features a considerar")
            params['max_features'] = None if max_features == "None" else max_features
        with col3:
            params['bootstrap'] = st.selectbox("Bootstrap", [True, False], help="Usar bootstrap al construir árboles")
        with col4:
            class_weight_option = st.selectbox("Class Weight", ["None", "balanced"], help="Pesos de clases")
            params['class_weight'] = None if class_weight_option == "None" else "balanced"

    elif algorithm == "LightGBM":
        st.markdown("**Parámetros de Boosting**")
        col1, col2, col3 = st.columns(3)
        with col1:
            params['n_estimators'] = st.slider("Número de Estimadores", 50, 500, 100, 10, help="Número de boosting rounds")
        with col2:
            params['learning_rate'] = st.slider("Tasa de Aprendizaje", 0.01, 0.3, 0.1, 0.01, help="Qué tan rápido aprende el modelo")
        with col3:
            params['num_leaves'] = st.slider("Número de Hojas", 20, 100, 31, 1, help="Complejidad del árbol")

        st.markdown("**Parámetros de Estructura**")
        col1, col2, col3 = st.columns(3)
        with col1:
            max_depth = st.slider("Profundidad Máxima", -1, 50, -1, 1, help="-1 significa sin límite")
            params['max_depth'] = max_depth
        with col2:
            params['min_child_samples'] = st.slider("Min Child Samples", 10, 100, 20, 5, help="Mínimo de datos en hoja")
        with col3:
            params['subsample'] = st.slider("Subsample", 0.5, 1.0, 0.8, 0.05, help="Fracción de datos para cada árbol")

        st.markdown("**Regularización**")
        col1, col2, col3 = st.columns(3)
        with col1:
            params['colsample_bytree'] = st.slider("Colsample by Tree", 0.5, 1.0, 0.8, 0.05, help="Fracción de features por árbol")
        with col2:
            params['reg_alpha'] = st.slider("Reg Alpha (L1)", 0.0, 1.0, 0.0, 0.05, help="Regularización L1")
        with col3:
            params['reg_lambda'] = st.slider("Reg Lambda (L2)", 0.0, 1.0, 0.0, 0.05, help="Regularización L2")

    st.markdown("---")

    if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
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

            # Guardar en session_state para usar después
            st.session_state['trained_model'] = model
            st.session_state['trained_le'] = le
            st.session_state['trained_X'] = X
            st.session_state['trained_X_test'] = X_test
            st.session_state['trained_y_test'] = y_test
            st.session_state['trained_y_pred_proba'] = y_pred_proba
            st.session_state['trained_accuracy'] = accuracy
            st.session_state['trained_report'] = report
            st.session_state['trained_cm'] = cm
            st.session_state['model_name'] = model_name
            st.session_state['algorithm'] = algorithm
            st.session_state['params'] = params

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

        # Feature Importance
        st.subheader("🎯 Importancia de Features (Feature Importance)")
        st.markdown("Muestra qué características tienen mayor impacto en las predicciones del modelo")

        if hasattr(model, 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)

            fig_imp = go.Figure(go.Bar(
                x=feature_imp['importance'],
                y=feature_imp['feature'],
                orientation='h',
                marker=dict(
                    color=feature_imp['importance'],
                    colorscale='Viridis',
                    showscale=True
                )
            ))

            fig_imp.update_layout(
                title="Top 20 Features Más Importantes",
                xaxis_title="Importancia",
                yaxis_title="Feature",
                height=500,
                template="plotly_dark",
                yaxis={'categoryorder': 'total ascending'}
            )

            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Este modelo no soporta feature importance directamente")

        st.markdown("---")

        # ROC Curves
        st.subheader("📈 Curvas ROC Multi-clase")
        st.markdown("Evalúa la capacidad del modelo para distinguir entre clases")

        col1, col2 = st.columns(2)

        with col1:
            # ROC Curves
            fig_roc = go.Figure()

            for i, class_name in enumerate(le.classes_):
                # One-vs-Rest
                y_test_binary = (y_test == i).astype(int)
                y_score = y_pred_proba[:, i]

                fpr, tpr, _ = roc_curve(y_test_binary, y_score)
                roc_auc = auc(fpr, tpr)

                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{class_name} (AUC = {roc_auc:.3f})',
                    line=dict(width=2)
                ))

            # Diagonal line
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random (AUC = 0.5)',
                line=dict(dash='dash', color='gray')
            ))

            fig_roc.update_layout(
                title="Curvas ROC por Clase",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=400,
                template="plotly_dark",
                hovermode='closest'
            )

            st.plotly_chart(fig_roc, use_container_width=True)

        with col2:
            # Precision-Recall Curves
            fig_pr = go.Figure()

            for i, class_name in enumerate(le.classes_):
                y_test_binary = (y_test == i).astype(int)
                y_score = y_pred_proba[:, i]

                precision, recall, _ = precision_recall_curve(y_test_binary, y_score)

                fig_pr.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name=f'{class_name}',
                    line=dict(width=2)
                ))

            fig_pr.update_layout(
                title="Curvas Precision-Recall por Clase",
                xaxis_title="Recall",
                yaxis_title="Precision",
                height=400,
                template="plotly_dark",
                hovermode='closest'
            )

            st.plotly_chart(fig_pr, use_container_width=True)

        st.markdown("---")

        # Análisis de errores
        st.subheader("🔍 Análisis de Errores del Modelo")
        st.markdown("Identifica dónde y por qué el modelo comete errores")

        errors_df = pd.DataFrame({
            'Real': le.inverse_transform(y_test),
            'Predicho': le.inverse_transform(y_pred),
            'Correcto': y_test == y_pred
        })

        col1, col2 = st.columns(2)

        with col1:
            # Error rate por clase
            error_by_class = []
            for cls in le.classes_:
                cls_mask = errors_df['Real'] == cls
                if cls_mask.sum() > 0:
                    error_rate = (~errors_df[cls_mask]['Correcto']).sum() / cls_mask.sum()
                    error_by_class.append({'Clase': cls, 'Error Rate': error_rate})

            error_df = pd.DataFrame(error_by_class)

            fig_errors = go.Figure(go.Bar(
                x=error_df['Clase'],
                y=error_df['Error Rate'],
                marker=dict(
                    color=error_df['Error Rate'],
                    colorscale='RdYlGn_r',
                    showscale=True
                ),
                text=[f'{e:.1%}' for e in error_df['Error Rate']],
                textposition='auto'
            ))

            fig_errors.update_layout(
                title="Tasa de Error por Clase",
                xaxis_title="Clase",
                yaxis_title="Error Rate",
                yaxis=dict(tickformat='.0%'),
                height=350,
                template="plotly_dark"
            )

            st.plotly_chart(fig_errors, use_container_width=True)

        with col2:
            # Confusiones más comunes
            st.markdown("**Confusiones Más Comunes**")
            errors_only = errors_df[~errors_df['Correcto']]
            if len(errors_only) > 0:
                confusion_counts = errors_only.groupby(['Real', 'Predicho']).size().reset_index(name='Count').sort_values('Count', ascending=False).head(10)

                st.dataframe(
                    confusion_counts.rename(columns={'Real': 'Clase Real', 'Predicho': 'Predicho Como', 'Count': 'Frecuencia'}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("¡No hay errores! Modelo perfecto en test set 🎉")

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

    # Opción para guardar modelo si ya hay uno entrenado en session_state
    if 'trained_model' in st.session_state:
        st.markdown("---")
        st.subheader("💾 Guardar Modelo Entrenado")
        st.markdown("Guarda el último modelo entrenado para usarlo en el módulo de Predicción")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"**Modelo:** {st.session_state['model_name']} | **Algoritmo:** {st.session_state['algorithm']} | **Accuracy:** {st.session_state['trained_accuracy']:.2%}")

        with col2:
            if st.button("💾 Guardar en Predictor", type="secondary", use_container_width=True, key="save_trained"):
                try:
                    # Crear directorio de modelos si no existe
                    os.makedirs('artifacts/saved_models', exist_ok=True)

                    # Preparar metadatos
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_filename = f"{st.session_state['model_name'].replace(' ', '_')}_{timestamp}"

                    # Guardar el modelo
                    model_path = f'artifacts/saved_models/{model_filename}.joblib'
                    joblib.dump(st.session_state['trained_model'], model_path)

                    # Guardar el LabelEncoder
                    le_path = f'artifacts/saved_models/{model_filename}_labelencoder.joblib'
                    joblib.dump(st.session_state['trained_le'], le_path)

                    # Guardar las columnas del modelo
                    columns_path = f'artifacts/saved_models/{model_filename}_columns.joblib'
                    joblib.dump(st.session_state['trained_X'].columns.tolist(), columns_path)

                    # Extraer métricas del reporte
                    report = st.session_state['trained_report']
                    avg_precision = report['weighted avg']['precision']
                    avg_recall = report['weighted avg']['recall']
                    avg_f1 = report['weighted avg']['f1-score']

                    # Guardar metadatos en JSON
                    metadata = {
                        'name': st.session_state['model_name'],
                        'algorithm': st.session_state['algorithm'],
                        'params': st.session_state['params'],
                        'accuracy': float(st.session_state['trained_accuracy']),
                        'avg_precision': float(avg_precision),
                        'avg_recall': float(avg_recall),
                        'avg_f1': float(avg_f1),
                        'timestamp': timestamp,
                        'model_file': model_filename,
                        'classes': st.session_state['trained_le'].classes_.tolist()
                    }

                    # Cargar o crear índice de modelos
                    index_path = 'artifacts/saved_models/models_index.json'
                    if os.path.exists(index_path):
                        with open(index_path, 'r') as f:
                            models_index = json.load(f)
                    else:
                        models_index = []

                    models_index.append(metadata)

                    with open(index_path, 'w') as f:
                        json.dump(models_index, f, indent=2)

                    st.success(f"✅ ¡Modelo guardado exitosamente como '{st.session_state['model_name']}'!")
                    st.balloons()
                    st.info("Ahora puedes usar este modelo en el Módulo de Predicción 🪐")

                except Exception as e:
                    st.error(f"❌ Error al guardar el modelo: {str(e)}")