import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Importamos nuestros módulos personalizados
# Nota: Asegúrate de ejecutar este script desde la raíz del proyecto para que encuentre 'src'
from src.data_loader import data_loader
from src.preprocessor import churn_preprocessor

class model_trainer:
    def __init__(self, data_path, model_output_path):
        self.data_path = data_path
        self.model_output_path = model_output_path
        self.pipeline = None

    def run(self):
        # 1. CARGAR DATOS
        print(">>> Paso 1: Cargando datos...")
        loader = data_loader(self.data_path)
        df = loader.load_data()

        # 2. SEPARAR FEATURES Y TARGET
        print(">>> Paso 2: Preparando features y target...")
        X = df.drop('Churn', axis=1)
        # Convertimos 'Yes'/'No' a 1/0
        y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

        # 3. SPLIT TRAIN/TEST
        # stratify=y es vital por el desbalance de clases
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 4. CONSTRUIR EL PIPELINE
        print(">>> Paso 3: Construyendo el Pipeline...")
        preprocessor = churn_preprocessor().make_preprocessor()
        
        # Aquí definimos el modelo Random Forest
        # class_weight='balanced' es el truco para arreglar el desbalance sin usar SMOTE externo
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced' 
        )

        # Unimos preprocesador + modelo en un solo objeto
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # 5. ENTRENAMIENTO
        print(">>> Paso 4: Entrenando el modelo (esto puede tardar un poco)...")
        self.pipeline.fit(X_train, y_train)

        # 6. EVALUACIÓN
        print(">>> Paso 5: Evaluando el modelo...")
        y_pred = self.pipeline.predict(X_test)
        
        print("\n--- REPORTE DE CLASIFICACIÓN ---")
        print(classification_report(y_test, y_pred))
        
        print("\n--- MATRIZ DE CONFUSIÓN ---")
        print(confusion_matrix(y_test, y_pred))

        print(f"\n--- ACCURACY: {accuracy_score(y_test, y_pred):.4f} ---")

        # 7. GUARDAR EL ARTEFACTO
        print(f"\n>>> Paso 6: Guardando el modelo en {self.model_output_path}...")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        
        # Guardamos el pipeline completo (incluye limpieza y modelo)
        joblib.dump(self.pipeline, self.model_output_path)
        print("¡Modelo guardado exitosamente!")

if __name__ == "__main__":
    # Configuración de rutas (ajusta según tu estructura de carpetas)
    DATA_PATH = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv' # Asegúrate que el nombre coincida con tu archivo
    MODEL_PATH = 'models/churn_model_v1.pkl'

    trainer = model_trainer(DATA_PATH, MODEL_PATH)
    trainer.run()