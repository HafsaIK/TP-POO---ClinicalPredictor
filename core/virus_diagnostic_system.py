"""
Système de diagnostic viral complet
Encapsule tous les composants du pipeline ML
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Union
from sklearn.preprocessing import StandardScaler
from core.model import ClinicalPredictor
from pipeline.evaluator import ModelEvaluator


class VirusDiagnosticSystem:
    """
    Système de diagnostic complet construit via le Builder pattern.
    Encapsule tous les composants nécessaires pour le diagnostic clinique.
    """
    
    def __init__(
        self,
        dataset: Any,
        scaler: StandardScaler,
        model: Any,
        predictor: ClinicalPredictor,
        evaluator: ModelEvaluator,
        metrics: Dict[str, float],
        X_test: np.ndarray,
        y_test: pd.Series
    ):
        """
        Initialise le système de diagnostic complet
        
        Args:
            dataset: Instance de ClinicalDataset
            scaler: StandardScaler utilisé pour la normalisation
            model: Modèle ML entraîné
            predictor: Instance de ClinicalPredictor
            evaluator: Instance de ModelEvaluator
            metrics: Dictionnaire des métriques de performance
            X_test: Données de test (features)
            y_test: Labels de test
        """
        self._dataset = dataset
        self._scaler = scaler
        self._model = model
        self._predictor = predictor
        self._evaluator = evaluator
        self._metrics = metrics
        self._X_test = X_test
        self._y_test = y_test
    
    def diagnose(self, patient_data: Union[np.ndarray, list]) -> str:
        """
        Effectue un diagnostic simple sur les données d'un patient
        
        Args:
            patient_data: Données du patient (features)
            
        Returns:
            Diagnostic: "Infecté" ou "Sain"
        """
        return self._predictor.diagnose(patient_data)
    
    def diagnose_with_probability(self, patient_data: Union[np.ndarray, list]) -> Dict[str, Any]:
        """
        Effectue un diagnostic avec probabilités
        
        Args:
            patient_data: Données du patient (features)
            
        Returns:
            Dictionnaire avec diagnostic, probabilité et confiance
        """
        return self._predictor.diagnose_proba(patient_data)
    
    def get_metrics(self) -> Dict[str, float]:
        """Retourne les métriques de performance du modèle"""
        return self._metrics.copy()
    
    def get_model(self) -> Any:
        """Retourne le modèle ML entraîné"""
        return self._model
    
    def get_predictor(self) -> ClinicalPredictor:
        """Retourne l'instance ClinicalPredictor"""
        return self._predictor
    
    def get_scaler(self) -> StandardScaler:
        """Retourne le StandardScaler utilisé"""
        return self._scaler
    
    def get_dataset(self) -> Any:
        """Retourne l'instance ClinicalDataset"""
        return self._dataset
    
    def get_test_data(self) -> tuple:
        """Retourne les données de test (X_test, y_test)"""
        return self._X_test, self._y_test
    
    def set_threshold(self, threshold: float):
        """
        Modifie le seuil de décision du predictor
        
        Args:
            threshold: Nouveau seuil (entre 0 et 1)
        """
        self._predictor.set_threshold(threshold)
    
    def evaluate_on_test_set(self) -> Dict[str, float]:
        """
        Ré-évalue le modèle sur les données de test
        
        Returns:
            Dictionnaire des métriques
        """
        return self._evaluator.evaluate(self._X_test, self._y_test)
    
    def __str__(self) -> str:
        """Représentation string du système"""
        return (
            f"VirusDiagnosticSystem(\n"
            f"  Model: {type(self._model).__name__}\n"
            f"  Accuracy: {self._metrics.get('accuracy', 0):.4f}\n"
            f"  F1-Score: {self._metrics.get('f1_score', 0):.4f}\n"
            f"  Test samples: {len(self._y_test)}\n"
            f")"
        )
    
    def __repr__(self) -> str:
        return self.__str__()
