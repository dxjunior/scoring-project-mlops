# Étape 2 — Préparation et traitement des données

## 1. Récupère le vrai dataset
1. Va sur https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
2. Télécharge le fichier `healthcare-dataset-stroke-data.csv`
3. Place-le dans `scoring_project/data/healthcare-dataset-stroke-data.csv`
   (remplace le fichier factice si tu en avais un)

## 2. Lance le script de nettoyage

```bash
python src\prepare_data.py
```
*(sous Mac/Linux : `python3 src/prepare_data.py`)*

## 3. Ce que fait le script, en détail

### a) Exploration initiale
Affiche les dimensions, types de colonnes, valeurs manquantes, et la
répartition de la variable cible `stroke`. Tu devrais voir un fort
déséquilibre (généralement ~95% de non-AVC contre ~5% d'AVC) — **c'est
normal et attendu**, on le traitera à l'Étape 4 (SMOTE, pondération).

### b) Nettoyage
- **`id`** supprimé (inutile pour la modélisation)
- **`bmi`** : valeurs manquantes imputées par la **médiane** (plus robuste
  que la moyenne face aux valeurs extrêmes)
- **`gender = 'Other'`** (très rare, 1 seule ligne dans le dataset
  original) fusionné avec `'Female'` pour éviter une catégorie non
  généralisable
- **`avg_glucose_level`** : outliers plafonnés (winsorisation via la
  méthode IQR) plutôt que supprimés — on ne veut pas perdre de cas
  positifs rares en supprimant des lignes

### c) Feature engineering (nouvelles variables créées)
| Feature | Description | Justification métier |
|---|---|---|
| `age_group` | Tranche d'âge (enfant/jeune adulte/adulte/senior) | Le risque d'AVC augmente fortement avec l'âge, non-linéairement |
| `cardio_risk_count` | Somme hypertension + maladie cardiaque | Score de risque cardiovasculaire cumulé |
| `bmi_category` | Catégorie IMC (sous-poids/normal/surpoids/obèse) | Seuils médicaux standards, plus interprétables qu'une valeur continue |
| `age_glucose_interaction` | Interaction âge × glucose | Ces deux facteurs combinés sont souvent plus prédictifs que séparément |

### d) Encodage
Toutes les variables catégorielles (`gender`, `work_type`,
`smoking_status`, etc.) sont encodées en one-hot (`pd.get_dummies`).

### e) Split train/test
80/20, **stratifié** sur la variable cible pour préserver la même
proportion de cas positifs dans les deux ensembles (important vu le
déséquilibre).

## 4. Fichiers générés
- `data/train_processed.csv`
- `data/test_processed.csv`

Ces fichiers sont prêts pour l'entraînement des modèles (Étape 4).

## ✅ Checklist de fin d'étape 2

- [ ] Vrai dataset Kaggle téléchargé et placé dans `data/`
- [ ] Script exécuté sans erreur
- [ ] Valeurs manquantes de `bmi` traitées
- [ ] Nouvelles features créées et visibles dans les fichiers de sortie
- [ ] `train_processed.csv` et `test_processed.csv` générés

## ➡️ Prochaine étape
Étape 3 : Définition du score métier — on va formaliser pourquoi un faux
négatif (dire "pas de risque" à un patient qui va faire un AVC) doit
coûter beaucoup plus cher qu'un faux positif, et transformer ça en une
métrique concrète à optimiser.
