import numpy as np
from typing import Any, Union

class ClinicalPredictor:
    """
    Classe principale pour la prédiction clinique
    Reçoit un modèle déjà entraîné et effectue des diagnostics
    """
    
    def __init__(self, trained_model: Any):
        
        self.model = trained_model
        self.threshold = 0.5
        
    def diagnose(self, patient_data: Union[np.ndarray, list]) -> str:
        """
        Effectue un diagnostic sur les données d'un patient
            "Infecté" si prédiction >= 0.5, "Sain" sinon
        """
        if isinstance(patient_data, list):
            patient_data = np.array(patient_data).reshape(1, -1)
        elif len(patient_data.shape) == 1:
            patient_data = patient_data.reshape(1, -1)
        
        prediction = self.model.predict(patient_data)[0]
        
        if prediction >= self.threshold:
            return "Infecté"
        else:
            return "Sain"
    
    def diagnose_proba(self, patient_data: Union[np.ndarray, list]) -> dict:
        
        if isinstance(patient_data, list):
            patient_data = np.array(patient_data).reshape(1, -1)
        elif len(patient_data.shape) == 1:
            patient_data = patient_data.reshape(1, -1)
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(patient_data)[0]
            prediction_value = proba[1]
        else:
            prediction_value = self.model.predict(patient_data)[0]
        
        diagnosis = "Infecté" if prediction_value >= self.threshold else "Sain"
        
        return {
            "diagnostic": diagnosis,
            "probabilite": float(prediction_value),
            "confiance": f"{prediction_value * 100:.2f}%"
        }
    
    def set_threshold(self, threshold: float):
        """Modifie le seuil de décision"""
        if not 0 <= threshold <= 1:
            raise ValueError("Le seuil doit être entre 0 et 1")
        self.threshold = threshold
