import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)

class ModelEvaluator:
    """Classe pour évaluer les modèles avec métriques intégrées"""
    
    def __init__(self, model):
        self.model = model
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calcule toutes les métriques principales
        
        Args:
            y_true: Vraies valeurs
            y_pred: Valeurs prédites
            
        Returns:
            Dictionnaire avec toutes les métriques
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        return metrics
    
    def print_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Affiche la matrice de confusion de manière formatée
        
        Args:
            y_true: Vraies valeurs
            y_pred: Valeurs prédites
        """
        cm = confusion_matrix(y_true, y_pred)
        print("\n📊 Matrice de Confusion:")
        print("    Prédit Sain | Prédit Malade")
        print(f"Réel Sain:     {cm[0][0]:3d}    |    {cm[0][1]:3d}")
        print(f"Réel Malade:   {cm[1][0]:3d}    |    {cm[1][1]:3d}")
        
    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Affiche le rapport de classification complet
        
        Args:
            y_true: Vraies valeurs
            y_pred: Valeurs prédites
        """
        print("\n📋 Rapport de Classification:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Sain', 'Diabétique'],
                                   zero_division=0))
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Évalue le modèle sur les données de test et affiche tous les résultats
        
        Args:
            X_test: Features de test
            y_test: Target de test
            
        Returns:
            Dictionnaire avec toutes les métriques
        """
        print("\n" + "="*60)
        print("📈 ÉVALUATION DU MODÈLE")
        print("="*60)
        
        y_pred = self.model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        print(f"\n🎯 Métriques de Performance:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        
        self.print_confusion_matrix(y_test, y_pred)
        self.print_classification_report(y_test, y_pred)
        
        return metrics