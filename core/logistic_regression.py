from sklearn.linear_model import LogisticRegression
import numpy as np


class LogisticRegressionModel:
    """Classe pour le modèle de régression logistique"""
    
    def __init__(self, random_state: int = 42, max_iter: int = 1010):
        """
        Initialise le modèle de régression logistique
        
        Args:
            random_state: Graine aléatoire pour la reproductibilité
            max_iter: Nombre maximum d'itérations
        """
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter
        )
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Entraîne le modèle
        
        Args:
            X_train: Données d'entraînement
            y_train: Étiquettes d'entraînement
        """
        print("⏳ Entraînement du modèle de régression logistique...")
        self.model.fit(X_train, y_train)
        print("✓ Entraînement terminé")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fait des prédictions
        
        Args:
            X: Données à prédire
            
        Returns:
            Prédictions du modèle
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne les probabilités prédites
        
        Args:
            X: Données à prédire
            
        Returns:
            Probabilités pour chaque classe
        """
        return self.model.predict_proba(X)
    
    def get_model(self):
        """Retourne le modèle entraîné"""
        return self.model
