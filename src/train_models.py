"""
Étape 4 - Entraînement et comparaison de modèles
==================================================
Ce script :
1. Définit une baseline simple (DummyClassifier)
2. Entraîne 3 modèles (Logistic Regression, Random Forest, XGBoost)
   avec GridSearchCV, optimisés sur le SCORE MÉTIER (pas l'accuracy)
3. Gère le déséquilibre des classes via SMOTE (sur-échantillonnage)
4. Calcule la feature importance (native + SHAP)
5. Compare tous les modèles entre eux et journalise tout dans MLFlow
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, recall_score,
    precision_score, confusion_matrix, ConfusionMatrixDisplay
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

from business_score import business_scorer, business_cost, business_score, find_optimal_threshold

# -----------------------------------------------------------------
# Configuration MLFlow
# -----------------------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("projet_scoring_modeles")

# -----------------------------------------------------------------
# 1. Chargement des données
# -----------------------------------------------------------------
train = pd.read_csv("data/train_processed.csv")
test = pd.read_csv("data/test_processed.csv")

X_train = train.drop(columns=["stroke"])
y_train = train["stroke"]
X_test = test.drop(columns=["stroke"])
y_test = test["stroke"]

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Taux de positifs (train): {y_train.mean():.4f}")

results_summary = []


def evaluate_and_log(model_name, model, X_test, y_test, params=None, log_shap=False):
    """Évalue un modèle entraîné, journalise dans MLFlow, retourne un résumé."""
    y_pred = model.predict(X_test)

    # Récupère les probabilités si possible pour chercher le seuil optimal
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        best_t, best_cost, _ = find_optimal_threshold(y_test, y_proba)
        y_pred_optimal = (y_proba >= best_t).astype(int)
    else:
        y_proba = None
        best_t = 0.5
        y_pred_optimal = y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "business_cost_default_threshold": business_cost(y_test, y_pred),
        "business_score_default_threshold": business_score(y_test, y_pred),
        "optimal_threshold": best_t,
        "business_cost_optimal_threshold": business_cost(y_test, y_pred_optimal),
        "business_score_optimal_threshold": business_score(y_test, y_pred_optimal),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    with mlflow.start_run(run_name=model_name):
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Matrice de confusion (seuil optimal)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_optimal, ax=ax)
        ax.set_title(f"{model_name} (seuil={best_t:.2f})")
        fig.savefig(f"confusion_matrix_{model_name}.png")
        mlflow.log_artifact(f"confusion_matrix_{model_name}.png")
        plt.close(fig)

        # Feature importance native si disponible
        final_estimator = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        if hasattr(final_estimator, "feature_importances_"):
            importances = pd.Series(
                final_estimator.feature_importances_, index=X_test.columns
            ).sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(8, 6))
            importances.plot(kind="barh", ax=ax)
            ax.set_title(f"Feature importance - {model_name}")
            ax.invert_yaxis()
            fig.tight_layout()
            fig.savefig(f"feature_importance_{model_name}.png")
            mlflow.log_artifact(f"feature_importance_{model_name}.png")
            plt.close(fig)

        # SHAP (optionnel, plus coûteux en calcul)
        if log_shap:
            try:
                explainer = shap.TreeExplainer(final_estimator)
                shap_values = explainer.shap_values(X_test)
                # Pour classification binaire, shap_values peut être une liste [neg, pos]
                sv = shap_values[1] if isinstance(shap_values, list) else shap_values
                fig = plt.figure()
                shap.summary_plot(sv, X_test, show=False, plot_type="bar")
                plt.tight_layout()
                plt.savefig(f"shap_summary_{model_name}.png")
                mlflow.log_artifact(f"shap_summary_{model_name}.png")
                plt.close(fig)
            except Exception as e:
                print(f"⚠️ SHAP a échoué pour {model_name}: {e}")

        # serialization_format="cloudpickle" : évite l'erreur de sécurité skops
        # qui bloque par défaut la sérialisation de pipelines imblearn (SMOTE)
        mlflow.sklearn.log_model(
            model, "model", serialization_format="cloudpickle"
        )

    print(f"\n--- {model_name} ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    results_summary.append({"model": model_name, **metrics})
    return metrics


# -----------------------------------------------------------------
# 2. Baseline
# -----------------------------------------------------------------
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
evaluate_and_log("baseline_dummy", baseline, X_test, y_test)

# -----------------------------------------------------------------
# 3. Logistic Regression (avec SMOTE + GridSearchCV)
# -----------------------------------------------------------------
pipe_lr = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
param_grid_lr = {
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ["l2"],
}
grid_lr = GridSearchCV(pipe_lr, param_grid_lr, scoring=business_scorer, cv=5, n_jobs=-1)
grid_lr.fit(X_train, y_train)
print(f"\nMeilleurs paramètres Logistic Regression : {grid_lr.best_params_}")
evaluate_and_log("logistic_regression", grid_lr.best_estimator_, X_test, y_test, grid_lr.best_params_)

# -----------------------------------------------------------------
# 4. Random Forest (avec SMOTE + GridSearchCV)
# -----------------------------------------------------------------
pipe_rf = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", RandomForestClassifier(random_state=42)),
])
param_grid_rf = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [5, 10, None],
}
grid_rf = GridSearchCV(pipe_rf, param_grid_rf, scoring=business_scorer, cv=5, n_jobs=-1)
grid_rf.fit(X_train, y_train)
print(f"\nMeilleurs paramètres Random Forest : {grid_rf.best_params_}")
evaluate_and_log("random_forest", grid_rf.best_estimator_, X_test, y_test, grid_rf.best_params_, log_shap=True)

# -----------------------------------------------------------------
# 5. XGBoost (avec SMOTE + GridSearchCV)
# -----------------------------------------------------------------
pipe_xgb = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", xgb.XGBClassifier(
        random_state=42, eval_metric="logloss", use_label_encoder=False
    )),
])
param_grid_xgb = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 5],
    "clf__learning_rate": [0.05, 0.1],
}
grid_xgb = GridSearchCV(pipe_xgb, param_grid_xgb, scoring=business_scorer, cv=5, n_jobs=-1)
grid_xgb.fit(X_train, y_train)
print(f"\nMeilleurs paramètres XGBoost : {grid_xgb.best_params_}")
evaluate_and_log("xgboost", grid_xgb.best_estimator_, X_test, y_test, grid_xgb.best_params_, log_shap=True)

# -----------------------------------------------------------------
# 6. Comparaison finale
# -----------------------------------------------------------------
summary_df = pd.DataFrame(results_summary)
summary_df = summary_df.sort_values("business_score_optimal_threshold", ascending=False)
summary_df.to_csv("model_comparison.csv", index=False)

print("\n" + "=" * 70)
print("TABLEAU COMPARATIF FINAL (trié par score métier au seuil optimal)")
print("=" * 70)
print(summary_df[["model", "business_score_optimal_threshold",
                   "business_cost_optimal_threshold", "recall", "precision", "roc_auc"]].to_string(index=False))

print(f"\n✅ Résultats sauvegardés dans model_comparison.csv")
print(f"👉 Lance 'mlflow ui --backend-store-uri sqlite:///mlflow.db' pour explorer tous les runs en détail")
