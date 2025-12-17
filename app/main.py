import os
import sys
import joblib
import streamlit as st

# Agregamos la raíz del proyecto al path para poder importar 'app.ui' si es necesario
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ui.main_window import MainWindow

# Constante de ruta del modelo
MODEL_PATH = 'models/churn_model_v1.pkl'

@st.cache_resource
def load_prediction_model():
    """
    Carga el modelo una sola vez y lo mantiene en memoria (Caché).
    """
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def main():
    # 1. Cargar el "Cerebro" (Modelo)
    pipeline = load_prediction_model()
    
    # 2. Verificar si el modelo cargó bien
    if pipeline is None:
        st.error("🚨 No se encontró el archivo del modelo.")
        st.warning(f"Por favor ejecuta: python -m src.train_model")
        st.stop() # Detiene la ejecución de la app

    # 3. Iniciar la "Cara" (Interfaz Gráfica)
    # Inyectamos el pipeline a la ventana principal
    app_window = MainWindow(pipeline)
    app_window.render()

if __name__ == "__main__":
    main()