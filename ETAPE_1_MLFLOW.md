# Étape 1 — Préparation de l'environnement MLFlow

## 1. Installation

Dans un terminal, à la racine de ton projet :

```bash
python3 -m venv venv
source venv/bin/activate      # sous Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Backend de suivi choisi : SQLite

Pour ce projet, on utilise **SQLite** comme backend de suivi. C'est un
excellent compromis pour un projet étudiant/portfolio :

- contrairement au stockage fichier basique, c'est une vraie base de
  données interrogeable ;
- contrairement à PostgreSQL, ça ne nécessite aucun serveur à installer
  ou configurer, tout tient dans un seul fichier `mlflow.db`.

Si tu veux passer à PostgreSQL plus tard (par exemple pour un déploiement
en équipe ou sur le cloud), il suffira de changer une seule ligne :

```python
# SQLite (actuel)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# PostgreSQL (plus tard, si besoin)
mlflow.set_tracking_uri("postgresql://user:password@host:5432/mlflow_db")
```

## 3. Tester la journalisation d'une expérience

Lance :

```bash
python3 src/test_mlflow_setup.py
```

Ce script :
- entraîne un `RandomForestClassifier` sur des données factices,
- log les **paramètres** (n_estimators, max_depth, random_state),
- log les **métriques** (accuracy, f1_score),
- log des **artefacts** (une matrice de confusion en image + le modèle
  entraîné complet).

Tu devrais voir s'afficher un message de succès avec un `Run ID`.
Deux nouveaux éléments apparaissent alors dans ton dossier : `mlflow.db`
(la base) et `mlruns/` ou `mlartifacts/` (les artefacts).

## 4. Visualiser les résultats dans l'interface MLFlow

Toujours dans le même dossier :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Puis ouvre **http://127.0.0.1:5000** dans ton navigateur. Tu devrais voir :
- l'expérience `projet_scoring_test_setup`,
- la run `test_random_forest` avec ses paramètres, métriques et
  artefacts (dont la matrice de confusion et le modèle sauvegardé).

## ✅ Checklist de fin d'étape 1

- [ ] MLFlow installé (`mlflow --version` fonctionne)
- [ ] Backend SQLite configuré et fonctionnel
- [ ] Une run test visible avec params + métriques + artefacts
- [ ] Interface MLFlow accessible et navigable dans le navigateur

## ➡️ Prochaine étape

Étape 2 : choisir un dataset Kaggle adapté à une classification binaire
et démarrer le nettoyage / feature engineering. Dis-moi quand tu es prêt
et je t'aiderai à choisir un bon dataset selon le domaine qui t'intéresse
(finance, santé, marketing...).
