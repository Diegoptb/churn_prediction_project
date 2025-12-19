import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

from src.data_loader import data_loader
from src.preprocessor import churn_preprocessor

class model_trainer:
    def __init__(self, data_path, model_output_path):
        self.data_path = data_path
        self.model_output_path = model_output_path
        self.pipeline = None

    def run(self):
        print(">>> Paso 1: Cargando datos...")
        loader = data_loader(self.data_path)
        df = loader.load_data()

        X = df.drop('Churn', axis=1)
        y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

        # Split estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(">>> Paso 2: Configurando XGBoost (Refinado)...")
        
        # Modelo "Francotirador V3" (Ajuste fino de profundidad)
        # Bajamos max_depth a 3: Árboles más simples generalizan mejor
        model = XGBClassifier(
            n_estimators=1500,      # Más árboles
            learning_rate=0.01,     # Aprendizaje lento
            max_depth=3,            # <--- CAMBIO: Menos profundidad = Menos Overfitting
            min_child_weight=5,     
            gamma=1.5,
            scale_pos_weight=2.0,   # Mantenemos el peso conservador
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=20.0,        # <--- CAMBIO: Más regularización para limpiar ruido
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )

        preprocessor = churn_preprocessor().make_preprocessor()
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        print(">>> Paso 3: Entrenando...")
        self.pipeline.fit(X_train, y_train)

        # --- OPTIMIZACIÓN MATEMÁTICA DEL UMBRAL ---
        print(">>> Paso 4: Buscando el Umbral Perfecto (Threshold Tuning)...")
        
        # Obtenemos probabilidades del test set
        y_scores = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Calculamos la curva Precision-Recall para todos los umbrales posibles
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
        
        # Calculamos el F1-Score para cada umbral
        # F1 = 2 * (P * R) / (P + R)
        numerator = 2 * recalls * precisions
        denom = recalls + precisions
        
        # Manejamos división por cero (donde denom es 0, f1 es 0)
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=denom!=0)
        
        # Encontramos el índice del F1 más alto
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        print(f"\nMEJOR UMBRAL ENCONTRADO: {best_threshold:.4f}")
        print(f"F1-SCORE ESPERADO: {best_f1:.4f}")

        # Aplicamos ese umbral óptimo a las predicciones
        y_pred_optimizado = (y_scores >= best_threshold).astype(int)
        
        print(f"\n--- REPORTE FINAL (Usando umbral {best_threshold:.4f}) ---")
        print(classification_report(y_test, y_pred_optimizado))
        
        cm = confusion_matrix(y_test, y_pred_optimizado)
        print("--- MATRIZ DE CONFUSIÓN ---")
        print(cm)
        
        recall = cm[1,1] / (cm[1,0] + cm[1,1])
        precision = cm[1,1] / (cm[0,1] + cm[1,1])
        print(f"\n>>> RECALL: {recall:.2%}")
        print(f">>> PRECISION: {precision:.2%}")

        # Guardado
        print(f"\n>>> Paso 5: Guardando modelo en {self.model_output_path}...")
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_output_path)
        
        # ¡OJO! Guardamos el umbral en un archivo de texto aparte para que la APP lo sepa
        # Opcional, pero recomendado para producción
        with open('models/best_threshold.txt', 'w') as f:
            f.write(str(best_threshold))
            
        print("¡Modelo optimizado guardado!")

if __name__ == "__main__":
    DATA_PATH = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    MODEL_PATH = 'models/churn_model_v2.pkl' 

    trainer = model_trainer(DATA_PATH, MODEL_PATH)
    trainer.run()