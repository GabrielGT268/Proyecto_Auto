import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.processed_data = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        try:
            self.data = pd.read_excel(self.file_path)
            print(f"Datos cargados exitosamente. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            return None
    
    def explore_data(self):
        if self.data is None:
            print("Primero debe cargar los datos")
            return

        print("=== INFO ===")
        print(f"Dimensiones: {self.data.shape}")
        print(f"Columnas disponibles: {len(self.data.columns)}")
        print(f"Valores faltantes por columna:")
        missing_data = self.data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            print(missing_data)
        else:
            print("No hay valores faltantes")
        
        return self.data.info()
    
    def clean_data(self):
        if self.data is None:
            print("Primero debe cargar los datos")
            return
        
        self.processed_data = self.data.copy()
        
        price_cols = ['MSRP', 'DealerCost']
        for col in price_cols:
            if col in self.processed_data.columns:
                self.processed_data[col] = (self.processed_data[col]
                    .astype(str)
                    .str.replace('$', '', regex=False)
                    .str.replace(',', '', regex=False)
                    .str.strip()
                    .replace('', np.nan))
                self.processed_data[col] = pd.to_numeric(self.processed_data[col], errors='coerce')
        
        categorical_cols = ['Brand', 'Model', 'VehicleClass', 'Region', 'DriveTrain']
        numerical_cols = ['MSRP', 'DealerCost', 'EngineSize', 'Cylinders', 'HorsePower',
                         'MPG_City', 'MPG_Highway', 'Weight', 'Wheelbase', 'Length']
        
        for col in numerical_cols:
            if col in self.processed_data.columns:
                self.processed_data[col] = self.processed_data[col].fillna(
                    self.processed_data[col].median()
                )
        
        for col in categorical_cols:
            if col in self.processed_data.columns:
                self.processed_data[col] = self.processed_data[col].fillna(
                    self.processed_data[col].mode()[0] if len(self.processed_data[col].mode()) > 0 else 'Unknown'
                )
        
        if 'MPG_City' in self.processed_data.columns and 'MPG_Highway' in self.processed_data.columns:
            self.processed_data['MPG_Average'] = (
                self.processed_data['MPG_City'] + self.processed_data['MPG_Highway']
            ) / 2
        
        if 'MSRP' in self.processed_data.columns and 'MPG_Average' in self.processed_data.columns:
            self.processed_data['Price_Efficiency_Ratio'] = (
                self.processed_data['MPG_Average'] / self.processed_data['MSRP'] * 100000
            )
        
        if 'MSRP' in self.processed_data.columns:
            self.processed_data['Price_Category'] = pd.cut(
                self.processed_data['MSRP'], 
                bins=[0, 20000, 35000, 50000, float('inf')],
                labels=['Económico', 'Medio', 'Premium', 'Lujo']
            )
        
        if 'MPG_Average' in self.processed_data.columns:
            self.processed_data['Efficiency_Category'] = pd.cut(
                self.processed_data['MPG_Average'],
                bins=[0, 20, 25, 30, float('inf')],
                labels=['Baja', 'Media', 'Alta', 'Muy Alta']
            )
        
        if 'HorsePower' in self.processed_data.columns and 'Weight' in self.processed_data.columns:
            self.processed_data['Power_to_Weight_Ratio'] = (
                self.processed_data['HorsePower'] / self.processed_data['Weight'] * 1000
            )
        
        if 'VehicleClass' in self.processed_data.columns:
            initial_count = len(self.processed_data)
            hybrid_count = len(self.processed_data[self.processed_data['VehicleClass'] == 'Hybrid'])
            self.processed_data = self.processed_data[self.processed_data['VehicleClass'] != 'Hybrid']
            final_count = len(self.processed_data)
            print(f"Vehículos híbridos excluidos del análisis: {hybrid_count}")
            print(f"Datos reducidos de {initial_count} a {final_count} vehículos")
        
        print("Datos limpiados exitosamente")
        return self.processed_data
    
    def encode_categorical_variables(self):
        if self.processed_data is None:
            print("Primero debe limpiar los datos")
            return
        
        categorical_cols = ['Brand', 'Model', 'VehicleClass', 'Region', 'DriveTrain']
        
        for col in categorical_cols:
            if col in self.processed_data.columns:
                le = LabelEncoder()
                self.processed_data[f'{col}_encoded'] = le.fit_transform(
                    self.processed_data[col].astype(str)
                )
                self.label_encoders[col] = le
        
        print("Variables categóricas codificadas")
        return self.processed_data
    
    def prepare_for_ml(self, target_column='MPG_Average'):
        if self.processed_data is None:
            print("Primero debe procesar los datos")
            return None, None
        
        feature_cols = []
        
        numerical_features = ['EngineSize', 'Cylinders', 'HorsePower', 'Weight', 'Wheelbase', 'Length']
        for col in numerical_features:
            if col in self.processed_data.columns:
                feature_cols.append(col)
        
        categorical_features = ['Brand_encoded', 'VehicleClass_encoded', 'Region_encoded', 'DriveTrain_encoded']
        for col in categorical_features:
            if col in self.processed_data.columns:
                feature_cols.append(col)
        
        X = self.processed_data[feature_cols].copy()
        y = self.processed_data[target_column].copy() if target_column in self.processed_data.columns else None
        
        if y is not None:
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
        else:
            X = X.dropna()
        
        print(f"Datos preparados para ML. Shape: X={X.shape}, y={y.shape if y is not None else 'None'}")
        return X, y
    
    def get_train_test_split(self, target_column='MPG_Average', test_size=0.2, random_state=42):
        X, y = self.prepare_for_ml(target_column)
        
        if X is None or y is None:
            return None, None, None, None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test

def main():
    processor = DataProcessor("../data/Proyecto.xlsx")
    
    data = processor.load_data()
    if data is not None:
        processor.explore_data()
        
        processor.clean_data()
        
        processor.encode_categorical_variables()
        
        X, y = processor.prepare_for_ml()
        
        if X is not None and y is not None:
            print(f"\nDATOS LISTOS.")
            print(f"Características: {X.columns.tolist()}")
            print(f"Variable objetivo: MPG_Average")

if __name__ == "__main__":
    main()
