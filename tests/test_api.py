"""
Tests automatisés de l'API de scoring
=======================================
Ces tests sont exécutés automatiquement par GitHub Actions à chaque
push/pull request, AVANT tout déploiement. Si un test échoue, le
déploiement est bloqué — c'est le principe même du CI/CD : on ne
déploie jamais du code cassé.

Lancer localement :
    pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


# -----------------------------------------------------------------
# Données de test réutilisables
# -----------------------------------------------------------------
VALID_PATIENT_HIGH_RISK = {
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

VALID_PATIENT_LOW_RISK = {
    "gender": "Female",
    "age": 25,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "No",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 85.0,
    "bmi": 22.0,
    "smoking_status": "never smoked",
}


# -----------------------------------------------------------------
# Tests des routes de base
# -----------------------------------------------------------------
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "message" in body


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


# -----------------------------------------------------------------
# Tests de l'endpoint /predict — cas valides
# -----------------------------------------------------------------
def test_predict_valid_high_risk_profile():
    response = client.post("/predict", json=VALID_PATIENT_HIGH_RISK)
    assert response.status_code == 200
    body = response.json()
    assert "stroke_probability" in body
    assert "stroke_risk" in body
    assert "threshold_used" in body
    assert "risk_level" in body
    assert 0.0 <= body["stroke_probability"] <= 1.0
    assert body["stroke_risk"] in [0, 1]
    assert body["risk_level"] in ["faible", "modéré", "élevé"]


def test_predict_valid_low_risk_profile():
    response = client.post("/predict", json=VALID_PATIENT_LOW_RISK)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["stroke_probability"] <= 1.0


def test_predict_response_structure():
    """Vérifie que la réponse contient exactement les champs attendus."""
    response = client.post("/predict", json=VALID_PATIENT_HIGH_RISK)
    body = response.json()
    expected_keys = {"stroke_probability", "stroke_risk", "threshold_used", "risk_level"}
    assert set(body.keys()) == expected_keys


# -----------------------------------------------------------------
# Tests de validation — cas invalides (doivent échouer proprement)
# -----------------------------------------------------------------
def test_predict_invalid_negative_age():
    payload = dict(VALID_PATIENT_HIGH_RISK)
    payload["age"] = -5
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Erreur de validation Pydantic


def test_predict_invalid_age_too_high():
    payload = dict(VALID_PATIENT_HIGH_RISK)
    payload["age"] = 200
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_gender():
    payload = dict(VALID_PATIENT_HIGH_RISK)
    payload["gender"] = "InvalidValue"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_required_field():
    payload = dict(VALID_PATIENT_HIGH_RISK)
    del payload["age"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_glucose():
    payload = dict(VALID_PATIENT_HIGH_RISK)
    payload["avg_glucose_level"] = -10
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# -----------------------------------------------------------------
# Test de cohérence métier
# -----------------------------------------------------------------
def test_high_risk_profile_scores_higher_than_low_risk():
    """
    Vérifie qu'un profil à facteurs de risque élevés (âge avancé,
    maladie cardiaque, glucose élevé) obtient une probabilité de risque
    supérieure à un profil jeune et sans facteur de risque connu.
    C'est un test de cohérence métier, pas juste technique.
    """
    response_high = client.post("/predict", json=VALID_PATIENT_HIGH_RISK)
    response_low = client.post("/predict", json=VALID_PATIENT_LOW_RISK)

    proba_high = response_high.json()["stroke_probability"]
    proba_low = response_low.json()["stroke_probability"]

    assert proba_high > proba_low
