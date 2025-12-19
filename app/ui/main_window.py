import streamlit as st
import pandas as pd

class MainWindow:
    def __init__(self, pipeline):
        """
        Recibe el pipeline (modelo) entrenado como dependencia.
        """
        self.pipeline = pipeline
        self.user_inputs = {}

    def render(self):
        """
        Método principal que 'dibuja' toda la interfaz.
        """
        self._setup_page()
        self._render_header()
        self._render_sidebar()
        self._render_main_form()

    def _setup_page(self):
        st.set_page_config(
            page_title="Telco Churn Predictor",
            page_icon="📊",
            layout="wide"
        )

    def _render_header(self):
        st.title("🔮 Predicción de Abandono de Clientes")
        st.markdown("""
        Sistema inteligente para la detección temprana de fuga de clientes (Churn).
        *Ingresa los datos del cliente a continuación:*
        """)

    def _render_sidebar(self):
        with st.sidebar:
            st.header("Configuración")
            st.info("Este panel simula la entrada de datos de un CRM.")
            st.write("Versión del Modelo: **v2.0 (XGBoost)**")

    def _render_main_form(self):
        # Dividimos el formulario en secciones lógicas
        with st.form("churn_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("👤 Demografía")
                self.user_inputs['gender'] = st.selectbox("Género", ["Male", "Female"])
                senior_citizen = st.selectbox("¿Es Jubilado?", ["No", "Yes"])
                self.user_inputs['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
                self.user_inputs['Partner'] = st.selectbox("¿Tiene Pareja?", ["Yes", "No"])
                self.user_inputs['Dependents'] = st.selectbox("¿Dependientes?", ["Yes", "No"])

            with col2:
                st.subheader("📡 Servicios")
                self.user_inputs['PhoneService'] = st.selectbox("Teléfono", ["Yes", "No"])
                self.user_inputs['MultipleLines'] = st.selectbox("Múltiples Líneas", ["No phone service", "No", "Yes"])
                self.user_inputs['InternetService'] = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
                self.user_inputs['OnlineSecurity'] = st.selectbox("Seguridad Online", ["No internet service", "No", "Yes"])
                self.user_inputs['TechSupport'] = st.selectbox("Soporte Técnico", ["No internet service", "No", "Yes"])
                # Rellenamos los que faltan con valores default para simplificar UI
                self.user_inputs['OnlineBackup'] = 'No'
                self.user_inputs['DeviceProtection'] = 'No'
                self.user_inputs['StreamingTV'] = 'No'
                self.user_inputs['StreamingMovies'] = 'No'

            with col3:
                st.subheader("💰 Contrato y Finanzas")
                self.user_inputs['Contract'] = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])
                self.user_inputs['PaperlessBilling'] = st.selectbox("Factura Digital", ["Yes", "No"])
                self.user_inputs['PaymentMethod'] = st.selectbox("Método de Pago", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                self.user_inputs['tenure'] = st.slider("Permanencia (Meses)", 0, 72, 12)
                self.user_inputs['MonthlyCharges'] = st.number_input("Cargo Mensual ($)", 0.0, 200.0, 70.0)
                # Calculamos TotalCharges automáticamente para facilitar la vida al usuario
                self.user_inputs['TotalCharges'] = self.user_inputs['tenure'] * self.user_inputs['MonthlyCharges']

            # Botón de envío del formulario
            submit_button = st.form_submit_button("Analizar Cliente", type="primary")

            if submit_button:
                self._process_prediction()

    def _process_prediction(self):
        # Convertir diccionario a DataFrame
        input_df = pd.DataFrame([self.user_inputs])
        
        # Validar que el modelo exista
        if not self.pipeline:
            st.error("Error crítico: El modelo no está cargado.")
            return

        try:
            # Obtener probabilidad
            probability = self.pipeline.predict_proba(input_df)[0][1]
            self._display_result(probability)
        except Exception as e:
            st.error(f"Error durante la predicción: {e}")

    def _display_result(self, probability):
            st.divider()
            st.subheader("📊 Resultados del Análisis")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            # Cargar umbral óptimo dinámicamente
            try:
                with open('models/best_threshold.txt', 'r') as f:
                    umbral_optimo = float(f.read())
            except:
                umbral_optimo = 0.45 # Fallback por si acaso
            
            with col_res1:
                st.metric(label="Probabilidad de Fuga", value=f"{probability:.1%}")
                st.caption(f"Umbral de decisión: {umbral_optimo:.1%}")
            
            with col_res2:
                if probability > umbral_optimo:
                    st.error("⚠️ ALERTA: RIESGO ALTO")
                    st.markdown(f"""
                    El modelo ha detectado patrones críticos de abandono.
                    **Probabilidad:** {probability:.3f} > **Umbral:** {umbral_optimo:.3f}
                    
                    **Acción recomendada:** Activar protocolo de retención inmediata.
                    """)
                else:
                    st.success("✅ CLIENTE SEGURO")
                    st.markdown(f"""
                    El cliente está por debajo del umbral de riesgo ({umbral_optimo:.3f}).
                    No se requieren acciones inmediatas.
                    """)