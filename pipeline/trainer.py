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
    
    @staticmethod
    def build_diagnostic_system(
        data_filepath: str,
        target_column: str = 'Outcome',
        model_type: str = 'random_forest',
        test_size: float = 0.2,
        random_state: int = 42,
        zero_replacement_columns: list = None,
        **model_params
    ):
        """
        Méthode factory pour créer un système de diagnostic complet
        en utilisant le VirusModelBuilder
        
        Args:
            data_filepath: Chemin vers le fichier de données CSV
            target_column: Nom de la colonne cible (défaut: 'Outcome')
            model_type: Type de modèle ('random_forest', 'logistic_regression', 'svm')
            test_size: Proportion des données de test (défaut: 0.2)
            random_state: Seed pour reproductibilité (défaut: 42)
            zero_replacement_columns: Liste des colonnes où remplacer les zéros
            **model_params: Paramètres additionnels pour le modèle
            
        Returns:
            VirusDiagnosticSystem: Système de diagnostic complet prêt à l'emploi
            
        Example:
            >>> system = ModelTrainer.build_diagnostic_system(
            ...     'data/clinical_data.csv',
            ...     model_type='random_forest',
            ...     n_estimators=100,
            ...     max_depth=10
            ... )
            >>> result = system.diagnose_with_probability(patient_data)
        """
        from pipeline.builder import VirusModelBuilder
        
        # Paramètres par défaut pour les colonnes zero_replacement si non fournis
        if zero_replacement_columns is None:
            zero_replacement_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        # Construire le système avec le Builder
        builder = VirusModelBuilder()
        
        system = (builder
            .set_data_source(data_filepath, target_column=target_column)
            .set_preprocessing_params(
                test_size=test_size,
                random_state=random_state,
                zero_replacement_columns=zero_replacement_columns
            )
            .set_model_type(model_type, **model_params)
            .build_and_train()
            .get_diagnostic_system()
        )
        
        return system