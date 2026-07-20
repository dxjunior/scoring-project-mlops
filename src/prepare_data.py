"""
Étape 2 - Préparation et traitement des données
=================================================
Dataset : Stroke Prediction Dataset (fedesoriano, Kaggle)
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Ce script :
1. Charge et explore les données brutes
2. Nettoie les valeurs manquantes et aberrantes
3. Encode les variables catégorielles
4. Crée de nouvelles features (feature engineering)
5. Sépare en train/test
6. Sauvegarde les jeux de données prêts à l'emploi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW_PATH = "data/healthcare-dataset-stroke-data.csv"
OUT_TRAIN = "data/train_processed.csv"
OUT_TEST = "data/test_processed.csv"

# -----------------------------------------------------------------
# 1. Chargement et exploration rapide
# -----------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
print("=" * 60)
print("EXPLORATION INITIALE")
print("=" * 60)
print(f"Dimensions : {df.shape}")
print(f"\nAperçu des colonnes et types :\n{df.dtypes}")
print(f"\nValeurs manquantes par colonne :\n{df.isnull().sum()}")
print(f"\nRépartition de la variable cible (stroke) :\n{df['stroke'].value_counts(normalize=True)}")

# -----------------------------------------------------------------
# 2. Nettoyage
# -----------------------------------------------------------------

# On retire l'identifiant, inutile pour la modélisation
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Traitement des valeurs manquantes de 'bmi' (variable numérique)
# On utilise la médiane plutôt que la moyenne, plus robuste aux outliers
bmi_median = df["bmi"].median()
n_missing_bmi = df["bmi"].isnull().sum()
df["bmi"] = df["bmi"].fillna(bmi_median)
print(f"\n✅ {n_missing_bmi} valeurs manquantes de 'bmi' imputées par la médiane ({bmi_median:.1f})")

# Catégorie 'Other' dans gender : très rare, on la fusionne avec 'Female'
# (choix documenté : évite une catégorie à 1-2 individus qui ne généralise pas)
if "gender" in df.columns:
    n_other = (df["gender"] == "Other").sum()
    if n_other > 0:
        df["gender"] = df["gender"].replace("Other", "Female")
        print(f"✅ {n_other} valeur(s) 'Other' dans gender fusionnée(s) avec 'Female'")

# Traitement des outliers sur avg_glucose_level (méthode IQR)
Q1 = df["avg_glucose_level"].quantile(0.25)
Q3 = df["avg_glucose_level"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
n_outliers = ((df["avg_glucose_level"] < lower_bound) | (df["avg_glucose_level"] > upper_bound)).sum()
# On plafonne (clip) plutôt que de supprimer, pour ne pas perdre de cas positifs rares
df["avg_glucose_level"] = df["avg_glucose_level"].clip(lower_bound, upper_bound)
print(f"✅ {n_outliers} outliers sur 'avg_glucose_level' plafonnés (winsorisation IQR)")

# -----------------------------------------------------------------
# 3. Feature engineering
# -----------------------------------------------------------------

# Catégories d'âge (les risques d'AVC augmentent fortement avec l'âge)
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 18, 40, 60, 100],
    labels=["child", "young_adult", "adult", "senior"]
)

# Indicateur de risque cardio cumulé (hypertension + maladie cardiaque)
df["cardio_risk_count"] = df["hypertension"] + df["heart_disease"]

# Indicateur BMI selon les catégories médicales standards
df["bmi_category"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 25, 30, 100],
    labels=["underweight", "normal", "overweight", "obese"]
)

# Interaction âge x glucose (les deux facteurs combinés sont souvent plus prédictifs)
df["age_glucose_interaction"] = df["age"] * df["avg_glucose_level"] / 100

print(f"\n✅ 4 nouvelles features créées : age_group, cardio_risk_count, bmi_category, age_glucose_interaction")

# -----------------------------------------------------------------
# 4. Encodage des variables catégorielles
# -----------------------------------------------------------------
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
print(f"\nColonnes catégorielles à encoder : {categorical_cols}")

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -----------------------------------------------------------------
# 5. Split train / test
# -----------------------------------------------------------------
X = df_encoded.drop(columns=["stroke"])
y = df_encoded["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = X_train.copy()
train_df["stroke"] = y_train
test_df = X_test.copy()
test_df["stroke"] = y_test

train_df.to_csv(OUT_TRAIN, index=False)
test_df.to_csv(OUT_TEST, index=False)

print("\n" + "=" * 60)
print("RÉSUMÉ FINAL")
print("=" * 60)
print(f"Train set : {train_df.shape} -> {OUT_TRAIN}")
print(f"Test set  : {test_df.shape} -> {OUT_TEST}")
print(f"Répartition stroke train : {y_train.value_counts(normalize=True).to_dict()}")
print(f"Répartition stroke test  : {y_test.value_counts(normalize=True).to_dict()}")
print(f"Nombre total de features après encodage : {X.shape[1]}")
