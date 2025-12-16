import pandas as pd

class data_loader:
    """
    Clase encargada de cargar y limpiar los datos crudos.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self) -> pd.DataFrame:
        print(f"Cargando datos desde {self.filepath}...")
        df = pd.read_csv(self.filepath)
        
        # Problema detectado en EDA: 'TotalCharges' tiene espacios vacíos " "
        # Reemplazamos espacio vacío con NaN y convertimos a float
        df['TotalCharges'] = df['TotalCharges'].replace(" ", 0)
        df['TotalCharges'] = df['TotalCharges'].astype(float)
                
        # Eliminamos customerID porque no sirve para predecir
        if 'customerID' in df.columns:
            df.drop(columns=['customerID'], inplace=True)
            
        print(f"Datos cargados exitosamente. Shape: {df.shape}")
        return df