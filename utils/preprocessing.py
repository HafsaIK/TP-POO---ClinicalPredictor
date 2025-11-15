import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class DataPreprocessor:
    """Classe pour le prétraitement des données"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Gère les valeurs manquantes
        
        Args:
            df: DataFrame
            strategy: 'mean', 'median', ou 'drop'
        """
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
        """
        Remplace les 0 biologiquement impossibles par la médiane
        Spécifique au dataset Pima Diabetes
        """
        X_copy = X.copy()
        for col in columns:
            if col in X_copy.columns:
                median_value = X_copy[X_copy[col] != 0][col].median()
                X_copy[col] = X_copy[col].replace(0, median_value)
        print(f"✓ Zéros remplacés pour {len(columns)} colonnes")
        return X_copy
    
    def normalize_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalise les features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("✓ Features normalisées")
        return X_train_scaled, X_test_scaled