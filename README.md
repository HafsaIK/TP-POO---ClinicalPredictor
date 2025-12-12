# ClinicalPredictor 🏥

![CI](https://github.com/HafsaIK/TP-POO---ClinicalPredictor/workflows/CI%20Pipeline/badge.svg)
![Tests](https://github.com/HafsaIK/TP-POO---ClinicalPredictor/workflows/Tests%20Pipeline/badge.svg)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Système de prédiction clinique pour le diagnostic du diabète utilisant l'apprentissage automatique.

## 📋 Description

ClinicalPredictor est un projet de génie logiciel qui implémente un système de diagnostic automatisé pour détecter le diabète à partir de données cliniques. Le projet utilise des techniques d'apprentissage automatique (Random Forest) pour prédire si un patient est diabétique ou sain.

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Installation des dépendances
```bash
pip install -r requirements.txt
```

## 📁 Structure du projet

```
TP-POO---ClinicalPredictor/
├── .github/
│   └── workflows/          # Workflows CI/CD
│       ├── ci.yml          # Pipeline CI (linting, qualité du code)
│       └── test.yml        # Pipeline de tests
├── core/                   # Modules principaux
│   ├── dataset.py          # Gestion des données cliniques
│   ├── model.py            # Modèle de prédiction
│   └── logistic_regression.py
├── pipeline/               # Pipeline ML
│   ├── trainer.py          # Entraînement des modèles
│   └── evaluator.py        # Évaluation des modèles
├── utils/                  # Utilitaires
│   ├── metrics.py          # Métriques d'évaluation
│   └── preprocessing.py    # Prétraitement des données
├── data/                   # Données cliniques
│   └── clinical_data.csv
├── tests/                  # Tests unitaires
│   └── test_imports.py
├── main.py                 # Point d'entrée principal
└── requirements.txt        # Dépendances Python
```

## 💻 Utilisation

### Exécution du programme principal
```bash
python main.py
```

Le programme va :
1. 📊 Charger les données cliniques
2. 🔧 Prétraiter les données (remplacement des zéros, normalisation)
3. 🎯 Entraîner le modèle Random Forest
4. 📈 Évaluer les performances du modèle
5. 🏥 Effectuer des diagnostics sur des patients de test

### Exécution des tests
```bash
# Tests unitaires
pytest

# Avec couverture de code
pytest --cov=. --cov-report=html

# Ouvrir le rapport de couverture
start htmlcov/index.html  # Windows
```

## 🔧 Workflows CI/CD

Le projet utilise GitHub Actions pour l'intégration continue :

### CI Pipeline (`ci.yml`)
- ✨ Vérification du formatage (Black)
- 📦 Organisation des imports (isort)
- 🔍 Analyse statique (flake8, pylint)
- 🔒 Analyse de sécurité (bandit)

### Tests Pipeline (`test.yml`)
- 🧪 Tests unitaires (pytest)
- 📊 Couverture de code
- 🔄 Tests d'intégration

Pour plus de détails, voir [.github/WORKFLOWS.md](.github/WORKFLOWS.md)

## 🛠️ Développement

### Installer les outils de développement
```bash
pip install pytest pytest-cov flake8 pylint black isort bandit
```

### Formater le code
```bash
# Auto-formatage avec Black
black .

# Organiser les imports
isort .
```

### Vérifier la qualité du code
```bash
# Linting
flake8 .
pylint **/*.py

# Sécurité
bandit -r .
```

## 📊 Métriques du modèle

Le modèle est évalué avec les métriques suivantes :
- **Accuracy** : Précision globale
- **Precision** : Précision par classe
- **Recall** : Rappel par classe
- **F1-Score** : Score F1 par classe
- **Matrice de confusion** : Visualisation des prédictions

## 🧪 Tests

```bash
# Exécuter tous les tests
pytest

# Tests avec verbose
pytest -v

# Tests avec couverture
pytest --cov=. --cov-report=html
```

## 📝 Fonctionnalités principales

- ✅ Chargement et exploration des données cliniques
- ✅ Prétraitement automatisé des données
- ✅ Entraînement de modèles ML (Random Forest)
- ✅ Évaluation complète des performances
- ✅ Système de diagnostic avec probabilités
- ✅ Tests automatisés
- ✅ CI/CD avec GitHub Actions

## 🤝 Contribution

1. Fork le projet
2. Créez votre branche (`git checkout -b feature/AmazingFeature`)
3. Commitez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Pushez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

**Important** : Assurez-vous que tous les tests passent et que le code est formaté avec Black avant de soumettre une PR.

## 📄 Licence

Ce projet est développé dans le cadre d'un TP de Génie Logiciel - Master S3.

## 👥 Auteurs

- **HafsaIK** - [GitHub](https://github.com/HafsaIK)

## 📚 Ressources

- [scikit-learn Documentation](https://scikit-learn.org/)
- [pandas Documentation](https://pandas.pydata.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
