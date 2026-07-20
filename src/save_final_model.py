"""
Étape 5a - Entraînement du modèle final pour déploiement
==========================================================
Random Forest a été retenu à l'Étape 4 (meilleur score métier).
Ce script réentraîne ce modèle avec ses meilleurs hyperparamètres sur
l'intégralité des données disponibles (train + test), et le sauvegarde
sous une forme légère et stable (joblib), indépendante de MLFlow, pour
qu'il soit facilement chargé par l'API.

On sauvegarde aussi la LISTE DES COLONNES attendues par le modèle après
encodage : c'est indispensable pour que l'API puisse reconstruire un
vecteur de features cohérent à partir d'une requête JSON brute.
"""

import json
import joblib
import pandas as pd

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from business_score import find_optimal_threshold

# -----------------------------------------------------------------
# 1. Chargement de TOUTES les données disponibles (train + test)
# -----------------------------------------------------------------
# Pour le modèle final en production, on utilise le maximum de données.
# (Le test set a déjà servi à valider le choix du modèle à l'Étape 4.)
train = pd.read_csv("data/train_processed.csv")
test = pd.read_csv("data/test_processed.csv")
full = pd.concat([train, test], ignore_index=True)

X = full.drop(columns=["stroke"])
y = full["stroke"]

print(f"Données complètes : {X.shape}, taux de positifs : {y.mean():.4f}")

# -----------------------------------------------------------------
# 2. Entraînement avec les meilleurs hyperparamètres trouvés (Étape 4)
# -----------------------------------------------------------------
# ⚠️ Remplace ces valeurs par TES meilleurs paramètres si différents
# de ceux obtenus dans ton propre GridSearchCV à l'Étape 4.
BEST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "random_state": 42,
}

pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", RandomForestClassifier(**BEST_PARAMS)),
])
pipeline.fit(X, y)

# -----------------------------------------------------------------
# 3. Détermination du seuil optimal final (sur le jeu de test Étape 4)
# -----------------------------------------------------------------
# On réutilise le test set pour fixer le seuil de décision en production
# (le modèle a été réentraîné sur tout, mais le seuil business reste
# celui validé précédemment pour rester cohérent avec l'Étape 3/4).
X_test = test.drop(columns=["stroke"])
y_test = test["stroke"]
y_proba_test = pipeline.predict_proba(X_test)[:, 1]
best_threshold, best_cost, _ = find_optimal_threshold(y_test, y_proba_test)
print(f"Seuil de décision retenu pour la production : {best_threshold:.2f}")

# -----------------------------------------------------------------
# 4. Sauvegarde du modèle + métadonnées nécessaires à l'API
# -----------------------------------------------------------------
joblib.dump(pipeline, "model_final.joblib")

metadata = {
    "model_name": "random_forest",
    "threshold": float(best_threshold),
    "feature_columns": list(X.columns),
    "params": BEST_PARAMS,
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Modèle sauvegardé : model_final.joblib")
print("✅ Métadonnées sauvegardées : model_metadata.json")
print(f"   -> {len(X.columns)} features attendues")
print(f"   -> seuil de décision : {best_threshold:.2f}")
