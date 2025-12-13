"""
Builder pour la construction du système de diagnostic viral
Implémente le design pattern Builder
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from core.dataset import ClinicalDataset
from core.model import ClinicalPredictor
from core.virus_diagnostic_system import VirusDiagnosticSystem
from pipeline.trainer import ModelTrainer
from pipeline.evaluator import ModelEvaluator


class VirusModelBuilder:
    """
    Builder pour construire un système de diagnostic viral complet.
    Construit étape par étape:
    1. Dataset
    2. Preprocessing
    3. Modèle ML
    4. Entraînement
    5. Évaluation
    6. Système de diagnostic final
    """
    
    def __init__(self):
        """Initialise le builder avec des valeurs par défaut"""
        # Configuration
        self._data_filepath: Optional[str] = None
        self._target_column: str = 'Outcome'
        self._test_size: float = 0.2
        self._random_state: int = 42
        
        # Preprocessing params
        self._zero_replacement_columns: list = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        self._missing_value_strategy: str = 'mean'
        
        # Model params
        self._model_type: str = 'random_forest'
        self._model_params: Dict[str, Any] = {}
        
        # Components construits
        self._dataset: Optional[ClinicalDataset] = None
        self._X_train: Optional[pd.DataFrame] = None
        self._X_test: Optional[pd.DataFrame] = None
        self._y_train: Optional[pd.Series] = None
        self._y_test: Optional[pd.Series] = None
        self._X_train_scaled: Optional[np.ndarray] = None
        self._X_test_scaled: Optional[np.ndarray] = None
        self._model: Optional[Any] = None
        self._metrics: Optional[Dict[str, float]] = None
        
    def set_data_source(self, filepath: str, target_column: str = 'Outcome') -> 'VirusModelBuilder':
        """
        Configure la source de données
        
        Args:
            filepath: Chemin vers le fichier CSV
            target_column: Nom de la colonne cible
            
        Returns:
            self pour le chaînage
        """
        self._data_filepath = filepath
        self._target_column = target_column
        print(f"✓ Source de données configurée: {filepath}")
        return self
    
    def set_preprocessing_params(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        zero_replacement_columns: Optional[list] = None,
        missing_value_strategy: str = 'mean'
    ) -> 'VirusModelBuilder':
        """
        Configure les paramètres de preprocessing
        
        Args:
            test_size: Proportion des données de test
            random_state: Seed pour la reproductibilité
            zero_replacement_columns: Colonnes où remplacer les 0
            missing_value_strategy: Stratégie pour les valeurs manquantes
            
        Returns:
            self pour le chaînage
        """
        self._test_size = test_size
        self._random_state = random_state
        if zero_replacement_columns is not None:
            self._zero_replacement_columns = zero_replacement_columns
        self._missing_value_strategy = missing_value_strategy
        print(f"✓ Paramètres de preprocessing configurés")
        return self
    
    def set_model_type(self, model_type: str, **model_params) -> 'VirusModelBuilder':
        """
        Configure le type de modèle et ses paramètres
        
        Args:
            model_type: 'random_forest', 'logistic_regression', ou 'svm'
            **model_params: Paramètres spécifiques au modèle
            
        Returns:
            self pour le chaînage
        """
        self._model_type = model_type
        self._model_params = model_params
        
        # Valeurs par défaut si non spécifiées
        if 'random_state' not in self._model_params:
            self._model_params['random_state'] = self._random_state
            
        print(f"✓ Type de modèle configuré: {model_type}")
        return self
    
    def build_and_train(self) -> 'VirusModelBuilder':
        """
        Construit et entraîne le système complet.
        Exécute toutes les étapes du pipeline:
        1. Chargement des données
        2. Séparation features/target
        3. Split train/test
        4. Preprocessing
        5. Entraînement
        6. Évaluation
        
        Returns:
            self pour le chaînage
        """
        if self._data_filepath is None:
            raise ValueError("La source de données doit être configurée d'abord (set_data_source)")
        
        print("\n" + "="*70)
        print("🔨 CONSTRUCTION DU SYSTÈME DE DIAGNOSTIC")
        print("="*70)
        
        # Étape 1: Chargement des données
        print("\n📊 ÉTAPE 1: CHARGEMENT DES DONNÉES")
        print("-" * 70)
        self._dataset = ClinicalDataset(self._data_filepath)
        data = self._dataset.load_data()
        print(f"🎯 Colonne cible: '{self._target_column}' (0=Sain, 1=Infecté)")
        
        # Étape 2: Séparation features/target
        self._dataset.split_features_target(target_column=self._target_column)
        
        # Étape 3: Split train/test
        print("\n🔧 ÉTAPE 2: PRÉTRAITEMENT DES DONNÉES")
        print("-" * 70)
        self._X_train, self._X_test, self._y_train, self._y_test = \
            self._dataset.get_train_test_split(
                test_size=self._test_size,
                random_state=self._random_state
            )
        print(f"✓ Train set: {self._X_train.shape[0]} patients")
        print(f"✓ Test set:  {self._X_test.shape[0]} patients")
        
        # Étape 4: Preprocessing
        # Remplacement des zéros biologiquement impossibles
        if self._zero_replacement_columns:
            self._X_train = self._dataset.replace_zeros(
                self._X_train,
                self._zero_replacement_columns
            )
            self._X_test = self._dataset.replace_zeros(
                self._X_test,
                self._zero_replacement_columns
            )
        
        # Normalisation
        self._X_train_scaled, self._X_test_scaled = \
            self._dataset.normalize_features(self._X_train, self._X_test)
        
        # Étape 5: Entraînement
        print("\n🎯 ÉTAPE 3: ENTRAÎNEMENT DU MODÈLE")
        print("-" * 70)
        trainer = ModelTrainer(model_type=self._model_type)
        trainer.create_model(**self._model_params)
        trainer.train(self._X_train_scaled, self._y_train)
        self._model = trainer.get_trained_model()
        
        # Étape 6: Évaluation
        print("\n📈 ÉTAPE 4: ÉVALUATION DU MODÈLE")
        print("-" * 70)
        evaluator = ModelEvaluator(self._model)
        self._metrics = evaluator.evaluate(self._X_test_scaled, self._y_test)
        
        print("\n" + "="*70)
        print("✅ CONSTRUCTION TERMINÉE AVEC SUCCÈS")
        print("="*70)
        
        return self
    
    def get_diagnostic_system(self) -> VirusDiagnosticSystem:
        """
        Retourne le système de diagnostic complet construit
        
        Returns:
            Instance de VirusDiagnosticSystem
            
        Raises:
            ValueError: Si le système n'a pas été construit
        """
        if self._model is None or self._metrics is None:
            raise ValueError(
                "Le système doit être construit d'abord (build_and_train)"
            )
        
        # Créer le predictor
        predictor = ClinicalPredictor(self._model)
        
        # Créer l'evaluator
        evaluator = ModelEvaluator(self._model)
        
        # Créer et retourner le système complet
        system = VirusDiagnosticSystem(
            dataset=self._dataset,
            scaler=self._dataset.scaler,
            model=self._model,
            predictor=predictor,
            evaluator=evaluator,
            metrics=self._metrics,
            X_test=self._X_test_scaled,
            y_test=self._y_test
        )
        
        print("\n🏥 Système de diagnostic prêt à l'emploi!")
        print(system)
        
        return system
    
    def reset(self) -> 'VirusModelBuilder':
        """
        Réinitialise le builder pour une nouvelle construction
        
        Returns:
            self pour le chaînage
        """
        self.__init__()
        return self
