from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

class ModelTrainer:
    """Classe pour entraîner les modèles"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Args:
            model_type: 'random_forest', 'logistic_regression', 'svm'
        """
        self.model_type = model_type
        self.model = None
        
    def create_model(self, **kwargs):
        """Crée le modèle selon le type spécifié"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                random_state=kwargs.get('random_state', 42),
                max_depth=kwargs.get('max_depth', 10)
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=kwargs.get('random_state', 42),
                max_iter=kwargs.get('max_iter', 1000)
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                random_state=kwargs.get('random_state', 42),
                probability=True
            )
        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}")
        
        print(f"✓ Modèle créé: {self.model_type}")
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Entraîne le modèle"""
        if self.model is None:
            raise ValueError("Le modèle doit être créé d'abord")
        
        print("⏳ Entraînement en cours...")
        self.model.fit(X_train, y_train)
        print("✓ Entraînement terminé")
        
    def get_trained_model(self):
        """Retourne le modèle entraîné"""
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné")
        return self.model