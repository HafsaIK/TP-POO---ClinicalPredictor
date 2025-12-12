"""Tests de base pour vérifier que les imports fonctionnent correctement"""
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_core_imports():
    """Vérifie que les modules core peuvent être importés sans erreur"""
    from core.dataset import ClinicalDataset
    from core.model import ClinicalPredictor
    
    assert ClinicalDataset is not None
    assert ClinicalPredictor is not None


def test_pipeline_imports():
    """Vérifie que les modules pipeline peuvent être importés sans erreur"""
    from pipeline.trainer import ModelTrainer
    from pipeline.evaluator import ModelEvaluator
    
    assert ModelTrainer is not None
    assert ModelEvaluator is not None


def test_utils_imports():
    """Vérifie que les modules utils peuvent être importés sans erreur"""
    import utils
    import utils.metrics
    import utils.preprocessing
    
    assert utils is not None
    assert utils.metrics is not None
    assert utils.preprocessing is not None


def test_main_import():
    """Vérifie que le fichier main peut être importé"""
    import main
    assert main is not None
    assert hasattr(main, 'main')
