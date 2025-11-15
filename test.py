import os

def create_project_structure():
    """Crée automatiquement toute la structure du projet"""
    
    # Dossiers à créer
    folders = ['data', 'core', 'pipeline', 'utils']
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✓ Dossier '{folder}/' créé")
    
    # Fichiers __init__.py à créer
    init_files = [
        'core/__init__.py',
        'pipeline/__init__.py',
        'utils/__init__.py'
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            pass  # Créer un fichier vide
        print(f"✓ Fichier '{init_file}' créé")
    
    print("\n✅ Structure du projet créée avec succès!")

if __name__ == "__main__":
    create_project_structure()