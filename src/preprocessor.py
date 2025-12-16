from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class churn_preprocessor:
    """
    Clase que configura el Pipeline de transformación de Scikit-Learn.
    """
    def __init__(self):
        self.numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_features = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 
            'PaperlessBilling', 'PaymentMethod'
        ]

    def make_preprocessor(self) -> ColumnTransformer:
        """
        Crea y retorna un ColumnTransformer listo para ser usado en un Pipeline.
        """
        # 1. Pipeline para variables numéricas
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # 2. Pipeline para variables categóricas
        categorical_transformer = Pipeline(steps=[
            # Rellena vacíos con 'missing' por seguridad
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            # OneHotEncoder con handle_unknown='ignore' para producción
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # 3. Unir ambos en un ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'  # Cualquier otra columna no listada se borra
        )
        
        return preprocessor