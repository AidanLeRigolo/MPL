# Neural_network
Petit dépôt d'exemples et d'expérimentations autour de réseaux neuronaux numériques en Python.

## Aperçu

Ce dépôt contient :

- `class_num_nn.py` : classes et fonctions pour construire un réseau neuronal dense simple (implémentation NumPy).
- `data_ia.py` : utilitaires pour la gestion / génération de données (jeu d'exemples).
- `nn_num/` : exemples et scripts numériques liés au réseau neuronal :
	- `class_num_nn.py` : (copie ou module réutilisable)
	- `test_pred_num.py` : script d'exécution / prédiction de démonstration

## Prérequis

Installer Python 3.10+ (ou celui de votre environnement). Le dépôt fonctionne avec un environnement virtuel.

Principales dépendances :

- `numpy`
- `scikit-learn` (importé comme `sklearn`)
- `scipy`
- `Pillow` (PIL)
- `matplotlib`

Recommandation : créer un environnement virtuel et installer les dépendances :

Windows PowerShell:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si vous ne disposez pas encore de `requirements.txt`, installez les paquets directement :

```powershell
python -m pip install numpy scikit-learn scipy pillow matplotlib
```

## Utilisation

Exemples rapides :

- Lancer un script de test/prédiction :

```powershell
.\\.venv\\Scripts\\python.exe nn_num\\test_pred_num.py
```

- Importer et réutiliser les classes du réseau :

```python
from class_num_nn import Layer_Dense
# construire un réseau, appeler .forward(), etc.
```

## Structure du dépôt

- `class_num_nn.py` — implémentation du réseau dense et fonctions associées.
- `data_ia.py` — génération et préparation de données d'entraînement/test.
- `nn_num/` — dossiers d'exemples et scripts.
- `README.md` — ce fichier.

## Tests et développement

Il n'y a pas de suite de tests automatisés fournie; pour tester rapidement les exemples, exécutez les scripts dans `nn_num/`.

## Fichiers recommandés

Ajoutez un fichier `requirements.txt` contenant :

```
numpy
scikit-learn
scipy
Pillow
matplotlib
```

## Contact

Pour toute question ou contribution, ouvrez une issue ou proposez une pull request.

---
Fait automatiquement par l'assistant — adaptez-le selon vos besoins.
# Neural_network