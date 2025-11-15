import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.dataset import ClinicalDataset
from core.model import ClinicalPredictor
from pipeline.trainer import ModelTrainer
from pipeline.evaluator import ModelEvaluator

def main():
    print("="*70)
    print("🏥 CLINICAL PREDICTOR - SYSTÈME DE DIAGNOSTIC DIABÈTE")
    print("="*70)
    
    # 1. Chargement des données
    print("\n📊 ÉTAPE 1: CHARGEMENT DES DONNÉES")
    print("-" * 70)
    dataset = ClinicalDataset('data/clinical_data.csv')
    data = dataset.load_data()
    
    target_column = 'Outcome'
    print(f"🎯 Colonne cible: '{target_column}' (0=Sain, 1=Diabétique)")
    
    dataset.split_features_target(target_column=target_column)
    
    # 2. Prétraitement
    print("\n🔧 ÉTAPE 2: PRÉTRAITEMENT DES DONNÉES")
    print("-" * 70)
    
    X_train, X_test, y_train, y_test = dataset.get_train_test_split(test_size=0.2)
    print(f"✓ Train set: {X_train.shape[0]} patients")
    print(f"✓ Test set:  {X_test.shape[0]} patients")
    
    # Remplacer les 0 biologiquement impossibles
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    X_train = dataset.replace_zeros(X_train, zero_columns)
    X_test = dataset.replace_zeros(X_test, zero_columns)
    
    # Normalisation
    X_train_scaled, X_test_scaled = dataset.normalize_features(X_train, X_test)
    
    # 3. Entraînement
    print("\n🎯 ÉTAPE 3: ENTRAÎNEMENT DU MODÈLE")
    print("-" * 70)
    trainer = ModelTrainer(model_type='random_forest')
    trainer.create_model(n_estimators=100, max_depth=10, random_state=42)
    trainer.train(X_train_scaled, y_train)
    
    # 4. Évaluation
    print("\n📈 ÉTAPE 4: ÉVALUATION DU MODÈLE")
    print("-" * 70)
    trained_model = trainer.get_trained_model()
    evaluator = ModelEvaluator(trained_model)
    metrics = evaluator.evaluate(X_test_scaled, y_test)
    
    # 5. Utilisation du ClinicalPredictor
    print("\n🏥 ÉTAPE 5: DIAGNOSTIC CLINIQUE")
    print("-" * 70)
    predictor = ClinicalPredictor(trained_model)
    
    # Test sur 5 patients aléatoires
    print("\n🔬 Test sur des patients du dataset de test:\n")
    for i in range(5):
        patient_data = X_test_scaled[i]
        real_diagnosis = "Diabétique" if y_test.iloc[i] == 1 else "Sain"
        
        diagnosis = predictor.diagnose(patient_data)
        diagnosis_proba = predictor.diagnose_proba(patient_data)
        
        print(f"Patient #{i+1}:")
        print(f"   Diagnostic prédit: {diagnosis_proba['diagnostic']}")
        print(f"   Confiance: {diagnosis_proba['confiance']}")
        print(f"   Diagnostic réel: {real_diagnosis}")
        print(f"   ✓ Correct" if diagnosis_proba['diagnostic'].lower() == real_diagnosis.lower() 
              else f"   ✗ Incorrect")
        print()
    
    # Test interactif (optionnel)
    print("\n" + "="*70)
    print("💡 Le système est prêt pour des diagnostics en temps réel!")
    print("="*70)
    print("\nExemple d'utilisation du ClinicalPredictor:")
    print(">>> patient = [6, 148, 72, 35, 0, 33.6, 0.627, 50]")
    print(">>> predictor.diagnose_proba(patient)")
    print("={'diagnostic': 'Infecté', 'probabilite': 0.75, 'confiance': '75.00%'}")

if __name__ == "__main__":
    main()