from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import numpy as np

class ModelMetrics:
    """Classe pour calculer les métriques du modèle"""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calcule toutes les métriques principales"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        return metrics
    
    @staticmethod
    def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
        """Affiche la matrice de confusion"""
        cm = confusion_matrix(y_true, y_pred)
        print("\n📊 Matrice de Confusion:")
        print("    Prédit Sain | Prédit Malade")
        print(f"Réel Sain:     {cm[0][0]:3d}    |    {cm[0][1]:3d}")
        print(f"Réel Malade:   {cm[1][0]:3d}    |    {cm[1][1]:3d}")
        
    @staticmethod
    def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
        """Affiche le rapport de classification"""
        print("\n📋 Rapport de Classification:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Sain', 'Diabétique'],
                                   zero_division=0))

