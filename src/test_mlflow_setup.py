"""
Étape 1 - Test de l'installation et de la configuration MLFlow
================================================================
Ce script vérifie que MLFlow est correctement installé et configuré
avec un backend de suivi (ici SQLite, facilement remplaçable par
PostgreSQL en changeant simplement l'URI de connexion).

Il entraîne un modèle jouet sur un dataset synthétique de classification
binaire, puis journalise :
- des paramètres (hyperparamètres du modèle)
- des métriques (accuracy, f1-score)
- un artefact (le modèle entraîné + un graphique)
"""

import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")  # pas d'affichage graphique nécessaire
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

# -----------------------------------------------------------------
# 1. Configuration du backend de suivi
# -----------------------------------------------------------------
# Backend de suivi SQLite : toutes les runs sont stockées dans une
# vraie base de données (mlflow.db), interrogeable et robuste.
# Pour passer à PostgreSQL plus tard, il suffira de remplacer l'URI par :
#   "postgresql://user:password@host:5432/mlflow_db"
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Nom de l'expérience (apparaîtra dans l'UI MLFlow)
mlflow.set_experiment("projet_scoring_test_setup")

# -----------------------------------------------------------------
# 2. Données factices (juste pour valider le pipeline MLFlow)
# -----------------------------------------------------------------
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    n_classes=2, weights=[0.7, 0.3], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------------
# 3. Entraînement + logging MLFlow
# -----------------------------------------------------------------
params = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42,
}

with mlflow.start_run(run_name="test_random_forest"):
    # --- log des paramètres ---
    mlflow.log_params(params)

    # --- entraînement ---
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- log des métriques ---
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # --- log d'un artefact graphique (matrice de confusion) ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    fig.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # --- log du modèle lui-même comme artefact MLFlow ---
    mlflow.sklearn.log_model(model, "model")

    print(f"✅ Run terminée avec succès.")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1-score: {f1:.4f}")
    print(f"   Run ID: {mlflow.active_run().info.run_id}")

print("\n👉 Pour visualiser les résultats, lance dans ton terminal (à la racine du projet) :")
print("   mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("   Puis ouvre http://127.0.0.1:5000 dans ton navigateur.")
