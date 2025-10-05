"""
Configuración compartida de la sidebar para todas las páginas de PlanetHunter
"""
import streamlit as st
import json
import os
import glob

def delete_model(model_file):
    """Elimina un modelo guardado y actualiza el índice"""
    try:
        # Eliminar archivos del modelo
        model_path = f'artifacts/saved_models/{model_file}.joblib'
        le_path = f'artifacts/saved_models/{model_file}_labelencoder.joblib'
        columns_path = f'artifacts/saved_models/{model_file}_columns.joblib'

        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(le_path):
            os.remove(le_path)
        if os.path.exists(columns_path):
            os.remove(columns_path)

        # Actualizar índice
        index_path = 'artifacts/saved_models/models_index.json'
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                models_index = json.load(f)

            # Filtrar el modelo eliminado
            models_index = [m for m in models_index if m.get('model_file') != model_file]

            with open(index_path, 'w') as f:
                json.dump(models_index, f, indent=2)

        return True
    except Exception as e:
        st.error(f"Error al eliminar modelo: {str(e)}")
        return False

def setup_sidebar():
    """Aplica el estilo y configuración de la sidebar de forma consistente"""

    # Cargar Font Awesome
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """, unsafe_allow_html=True)

    # CSS completamente estático - se carga inmediatamente sin parpadeo
    st.markdown("""
    <style>
        /* Contenedor de navegación con espacio para logo y título */
        [data-testid="stSidebarNav"] {
            padding-top: 180px;
            position: relative;
        }

        /* Logo circular ARRIBA */
        [data-testid="stSidebarNav"]::before {
            content: "\\f57d";
            font-family: "Font Awesome 6 Free";
            font-weight: 900;
            display: flex;
            align-items: center;
            justify-content: center;

            /* Posición absoluta arriba */
            position: absolute;
            top: 0.5rem;
            left: 50%;
            transform: translateX(-50%);

            /* Estilo del círculo */
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            color: white;
            font-size: 2.5rem;
        }

        /* Título PlanetHunter ARRIBA (debajo del logo) */
        [data-testid="stSidebarNav"]::after {
            content: "PlanetHunter";
            display: block;
            text-align: center;
            font-family: "Source Sans Pro", sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: #f1f5f9;

            /* Posición absoluta arriba */
            position: absolute;
            top: 95px;
            left: 0;
            right: 0;
            padding-bottom: 1rem;
            border-bottom: 1px solid #334155;
            margin: 0 1rem 1rem 1rem;
        }

        /* Ocultar texto original de app.py INMEDIATAMENTE */
        [data-testid="stSidebarNav"] li:first-child a span {
            visibility: hidden;
            position: absolute;
            width: 0;
            height: 0;
            opacity: 0;
        }

        /* Reemplazar con Home + icono */
        [data-testid="stSidebarNav"] li:first-child a::before {
            font-family: "Font Awesome 6 Free";
            font-weight: 900;
        }

        [data-testid="stSidebarNav"] li:first-child a::after {
            content: "Home";
            font-family: "Source Sans Pro", sans-serif;
        }

        /* Estilo consistente para TODOS los botones de navegación */
        [data-testid="stSidebarNav"] a {
            font-family: "Source Sans Pro", sans-serif !important;
            font-size: 1rem !important;
            font-weight: 400 !important;
            padding: 0.5rem 1rem !important;
            display: flex !important;
            align-items: center !important;
        }

        /* Ajustar espaciado de la lista de navegación */
        [data-testid="stSidebarNav"] ul {
            padding-top: 0.5rem;
            margin-top: 0;
        }

        /* Hover effect consistente */
        [data-testid="stSidebarNav"] a:hover {
            background-color: rgba(99, 102, 241, 0.1) !important;
        }
    </style>
    """, unsafe_allow_html=True)

def show_model_manager():
    """Muestra el componente de administración de modelos en el contenido principal"""
    with st.expander("⚙️ Administrador de Modelos"):
        st.markdown("### Gestiona tus modelos guardados")

        # Cargar modelos guardados
        index_path = 'artifacts/saved_models/models_index.json'

        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                saved_models = json.load(f)

            if saved_models:
                st.info(f"📊 Total de modelos guardados: **{len(saved_models)}**")

                # Tabla de modelos
                for idx, model in enumerate(saved_models):
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])

                        with col1:
                            st.markdown(f"**{model.get('name', 'Sin nombre')}**")

                        with col2:
                            st.text(f"{model.get('algorithm', 'N/A')}")

                        with col3:
                            st.metric("Accuracy", f"{model.get('accuracy', 0):.1%}", label_visibility="collapsed")

                        with col4:
                            st.metric("F1", f"{model.get('avg_f1', 0):.1%}", label_visibility="collapsed")

                        with col5:
                            if st.button("🗑️ Eliminar", key=f"del_{model.get('model_file')}", type="secondary", use_container_width=True):
                                if delete_model(model.get('model_file')):
                                    st.success("✓ Modelo eliminado correctamente")
                                    st.rerun()

                        st.divider()
            else:
                st.warning("No hay modelos guardados. Entrena y guarda modelos desde **Model Workshop**.")
        else:
            st.warning("No hay modelos guardados. Entrena y guarda modelos desde **Model Workshop**.")
