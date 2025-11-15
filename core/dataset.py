
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ClinicalDataset:
    """Classe pour gérer le dataset clinique avec prétraitement intégré"""
    
    def __init__(self, filepath: str):
        
        self.filepath = filepath
        self.data = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        
    def load_data(self) -> pd.DataFrame:
        self.data = pd.read_csv(self.filepath)
        print(f"✓ Données chargées: {self.data.shape}")
        return self.data
    
    def split_features_target(self, target_column: str):
       
        if self.data is None:
            raise ValueError("Les données doivent être chargées d'abord")
        
        self.X = self.data.drop(columns=[target_column])
        self.y = self.data[target_column]
        
        print(f"✓ Features: {self.X.shape}, Target: {self.y.shape}")
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
       
        df_copy = df.copy()
        
        if strategy == 'drop':
            df_copy = df_copy.dropna()
        else:
            for col in df_copy.select_dtypes(include=[np.number]).columns:
                if df_copy[col].isnull().any():
                    if strategy == 'mean':
                        df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                    elif strategy == 'median':
                        df_copy[col].fillna(df_copy[col].median(), inplace=True)
        
        print(f"✓ Valeurs manquantes traitées (stratégie: {strategy})")
        return df_copy
    
    def replace_zeros(self, X: pd.DataFrame, columns: list) -> pd.DataFrame:
        
        X_copy = X.copy()
        for col in columns:
            if col in X_copy.columns:
                median_value = X_copy[X_copy[col] != 0][col].median()
                X_copy[col] = X_copy[col].replace(0, median_value)
        print(f"✓ Zéros remplacés pour {len(columns)} colonnes")
        return X_copy
    
    def normalize_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
       
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("✓ Features normalisées")
        return X_train_scaled, X_test_scaled
        
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
       
        if self.X is None or self.y is None:
            raise ValueError("Features et target doivent être définis")
        
        return train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state
        )