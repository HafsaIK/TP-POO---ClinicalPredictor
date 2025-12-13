import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.builder import VirusModelBuilder
from pipeline.trainer import ModelTrainer


def main():
    print("="*70)
    print("🏥 CLINICAL PREDICTOR - SYSTÈME DE DIAGNOSTIC DIABÈTE")
    print("="*70)
    print("\n✨ Démonstration du Design Pattern Builder\n")
    
    # =========================================================================
    # MÉTHODE 1 : Utilisation de la méthode factory (RECOMMANDÉE - Plus simple)
    # =========================================================================
    print("📌 MÉTHODE 1 : Utilisation de la méthode factory ModelTrainer")
    print("="*70)
    print("Code utilisé :")
    print(">>> system = ModelTrainer.build_diagnostic_system(")
    print("...     'data/clinical_data.csv',")
    print("...     model_type='random_forest',")
    print("...     n_estimators=100,")
    print("...     max_depth=10")
    print("... )")
    print()
    
    # Construction simple avec la méthode factory
    system = ModelTrainer.build_diagnostic_system(
        'data/clinical_data.csv',
        model_type='random_forest',
        n_estimators=100,
        max_depth=10
    )
    
    # =========================================================================
    # MÉTHODE 2 : Utilisation directe du Builder (Pour plus de contrôle)
    # =========================================================================
    print("\n📌 MÉTHODE 2 : Utilisation directe du VirusModelBuilder")
    print("="*70)
    print("Code utilisé :")
    print(">>> system2 = (VirusModelBuilder()")
    print("...     .set_data_source('data/clinical_data.csv')")
    print("...     .set_preprocessing_params(test_size=0.2)")
    print("...     .set_model_type('logistic_regression', max_iter=1000)")
    print("...     .build_and_train()")
    print("...     .get_diagnostic_system()")
    print("... )")
    print()
    
    # Construction avec le Builder pour comparaison
    system2 = (VirusModelBuilder()
        .set_data_source('data/clinical_data.csv', target_column='Outcome')
        .set_preprocessing_params(test_size=0.2, random_state=42)
        .set_model_type('logistic_regression', max_iter=1000)
        .build_and_train()
        .get_diagnostic_system()
    )
    
    # =========================================================================
    # UTILISATION DU SYSTÈME DE DIAGNOSTIC
    # =========================================================================
    print("\n🏥 DIAGNOSTIC CLINIQUE")
    print("="*70)
    
    # Récupérer les données de test
    X_test_scaled, y_test = system.get_test_data()
    
    # Test sur 5 patients aléatoires
    print("\n🔬 Test sur des patients du dataset de test:\n")
    for i in range(5):
        patient_data = X_test_scaled[i]
        real_diagnosis = "Diabétique" if y_test.iloc[i] == 1 else "Sain"
        
        diagnosis_proba = system.diagnose_with_probability(patient_data)
        
        print(f"Patient #{i+1}:")
        print(f"   Diagnostic prédit: {diagnosis_proba['diagnostic']}")
        print(f"   Confiance: {diagnosis_proba['confiance']}")
        print(f"   Diagnostic réel: {real_diagnosis}")
        print(f"   ✓ Correct" if diagnosis_proba['diagnostic'].lower() == real_diagnosis.lower() 
              else f"   ✗ Incorrect")
        print()
    
    # =========================================================================
    # COMPARAISON DES DEUX SYSTÈMES
    # =========================================================================
    print("\n📊 COMPARAISON DES MÉTRIQUES")
    print("="*70)
    
    metrics1 = system.get_metrics()
    metrics2 = system2.get_metrics()
    
    print(f"\n{'Méthode':<30} {'Model':<20} {'Accuracy':<12} {'F1-Score':<12}")
    print("-" * 70)
    print(f"{'Factory Method':<30} {'Random Forest':<20} {metrics1['accuracy']:<12.4f} {metrics1['f1_score']:<12.4f}")
    print(f"{'Builder Direct':<30} {'Log. Regression':<20} {metrics2['accuracy']:<12.4f} {metrics2['f1_score']:<12.4f}")
    
    # =========================================================================
    # RÉSUMÉ ET RECOMMANDATIONS
    # =========================================================================
    print("\n" + "="*70)
    print("💡 RÉSUMÉ DES DEUX APPROCHES")
    print("="*70)
    
    print("\n✨ Méthode Factory (ModelTrainer.build_diagnostic_system):")
    print("   ✓ Plus simple - une seule ligne")
    print("   ✓ Paramètres par défaut intelligents")
    print("   ✓ Idéal pour les cas d'usage standards")
    print("   ✓ Recommandé pour débutants")
    
    print("\n🔧 Builder Direct (VirusModelBuilder):")
    print("   ✓ Contrôle total sur chaque étape")
    print("   ✓ Configuration personnalisée")
    print("   ✓ Idéal pour cas complexes")
    print("   ✓ Recommandé pour utilisateurs avancés")
    
    print("\n📝 Exemple d'utilisation recommandée:")
    print("   >>> from pipeline.trainer import ModelTrainer")
    print("   >>> system = ModelTrainer.build_diagnostic_system(")
    print("   ...     'data/clinical_data.csv',")
    print("   ...     model_type='random_forest',")
    print("   ...     n_estimators=100")
    print("   ... )")
    print("   >>> result = system.diagnose_with_probability(patient_data)")
    print("   >>> print(result)")


if __name__ == "__main__":
    main()
