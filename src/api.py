"""
Étape 5b - API de scoring avec FastAPI
========================================
Cette API reçoit les données BRUTES d'un patient (pas encore encodées),
applique exactement le même feature engineering que l'Étape 2, aligne
les colonnes sur celles attendues par le modèle, puis retourne :
- la probabilité de risque d'AVC
- la décision finale (0/1) selon le seuil métier optimisé (Étape 3/4)

Lancer localement :
    uvicorn src.api:app --reload --port 8000

Documentation interactive auto-générée disponible sur :
    http://127.0.0.1:8000/docs
"""

import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

# -----------------------------------------------------------------
# Chargement du modèle et des métadonnées au démarrage de l'API
# -----------------------------------------------------------------
MODEL_PATH = "model_final.joblib"
METADATA_PATH = "model_metadata.json"

model = joblib.load(MODEL_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)

FEATURE_COLUMNS = metadata["feature_columns"]
THRESHOLD = metadata["threshold"]

app = FastAPI(
    title="API de Scoring - Risque d'AVC",
    description="Prédit le risque d'AVC (stroke) à partir de données patient.",
    version="1.0.0",
)


# -----------------------------------------------------------------
# Schéma de la requête (données brutes, telles que saisies par un
# utilisateur ou un formulaire, AVANT tout feature engineering)
# -----------------------------------------------------------------
class PatientData(BaseModel):
    gender: Literal["Male", "Female"]
    age: float = Field(..., ge=0, le=120, description="Âge en années")
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    ever_married: Literal["Yes", "No"]
    work_type: Literal["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    Residence_type: Literal["Urban", "Rural"]
    avg_glucose_level: float = Field(..., ge=0, description="Niveau moyen de glucose (mg/dL)")
    bmi: float = Field(..., ge=0, description="Indice de masse corporelle")
    smoking_status: Literal["never smoked", "formerly smoked", "smokes", "Unknown"]

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "age": 67,
                "hypertension": 0,
                "heart_disease": 1,
                "ever_married": "Yes",
                "work_type": "Private",
                "Residence_type": "Urban",
                "avg_glucose_level": 228.69,
                "bmi": 36.6,
                "smoking_status": "formerly smoked",
            }
        }


class PredictionResponse(BaseModel):
    stroke_probability: float
    stroke_risk: int
    threshold_used: float
    risk_level: str


def preprocess(patient: PatientData) -> pd.DataFrame:
    """
    Reproduit EXACTEMENT le même feature engineering que
    src/prepare_data.py (Étape 2), pour une seule ligne de données.
    """
    df = pd.DataFrame([patient.model_dump()])

    # --- Feature engineering (identique à l'Étape 2) ---
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 18, 40, 60, 100],
        labels=["child", "young_adult", "adult", "senior"]
    )
    df["cardio_risk_count"] = df["hypertension"] + df["heart_disease"]
    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"]
    )
    df["age_glucose_interaction"] = df["age"] * df["avg_glucose_level"] / 100

    # --- Encodage one-hot ---
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # --- Alignement STRICT sur les colonnes attendues par le modèle ---
    # Toute colonne manquante (catégorie non présente dans cette requête)
    # est ajoutée avec la valeur 0 ; l'ordre est celui de l'entraînement.
    df_aligned = df_encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return df_aligned


@app.get("/")
def root():
    return {
        "message": "API de scoring - Risque d'AVC",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    try:
        X = preprocess(patient)
        proba = float(model.predict_proba(X)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

    risk = int(proba >= THRESHOLD)
    if proba < 0.15:
        risk_level = "faible"
    elif proba < THRESHOLD:
        risk_level = "modéré"
    else:
        risk_level = "élevé"

    return PredictionResponse(
        stroke_probability=round(proba, 4),
        stroke_risk=risk,
        threshold_used=THRESHOLD,
        risk_level=risk_level,
    )
