# Étape 5 — Déploiement de l'API de scoring

## 1. Installer les nouvelles dépendances

```bash
pip install -r requirements.txt
```

## 2. Générer le modèle final pour la production

```bash
python src\save_final_model.py
```

Ce script :
- réentraîne Random Forest (le modèle retenu à l'Étape 4) avec ses
  meilleurs hyperparamètres, sur **toutes** les données disponibles
- détermine le seuil de décision optimal
- sauvegarde deux fichiers à la racine du projet :
  - **`model_final.joblib`** : le modèle entraîné, prêt à charger
  - **`model_metadata.json`** : le seuil retenu + la liste exacte des
    colonnes attendues (indispensable pour que l'API reconstruise les
    bonnes features)

⚠️ **Vérifie tes hyperparamètres** : dans le script, la variable
`BEST_PARAMS` contient ceux obtenus dans MON test. Remplace-les par
**tes propres** meilleurs paramètres trouvés à l'Étape 4 (visibles dans
le tableau `Meilleurs paramètres Random Forest : {...}` affiché par
`train_models.py`).

## 3. Lancer l'API en local

```bash
uvicorn src.api:app --reload --port 8000
```

Puis ouvre **http://127.0.0.1:8000/docs** dans ton navigateur : FastAPI
génère automatiquement une interface de test interactive (Swagger UI).
Tu peux y tester directement l'endpoint `/predict` avec l'exemple
pré-rempli, sans écrire une seule ligne de code.

## 4. Comment fonctionne l'API

### Endpoints
- **`GET /`** : message d'accueil
- **`GET /health`** : vérifie que l'API et le modèle sont bien chargés
- **`POST /predict`** : reçoit les données brutes d'un patient, renvoie
  la prédiction

### Exemple de requête (via `/docs` ou `curl`)
```json
{
  "gender": "Male",
  "age": 67,
  "hypertension": 0,
  "heart_disease": 1,
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "smoking_status": "formerly smoked"
}
```

### Exemple de réponse
```json
{
  "stroke_probability": 0.62,
  "stroke_risk": 1,
  "threshold_used": 0.34,
  "risk_level": "élevé"
}
```

### Point technique important : cohérence du feature engineering
L'API **reproduit exactement** les mêmes transformations que le script
de l'Étape 2 (`age_group`, `cardio_risk_count`, `bmi_category`,
`age_glucose_interaction`, encodage one-hot), puis **aligne** les
colonnes sur celles vues à l'entraînement (`model_metadata.json`). Sans
cet alignement strict, le modèle recevrait des colonnes dans le
mauvais ordre ou manquantes → prédictions fausses ou erreurs silencieuses.
**C'est un piège classique en mise en production, à mentionner dans ton
rapport.**

### Validation automatique des entrées
Grâce à Pydantic (utilisé par FastAPI), toute donnée invalide est
rejetée automatiquement avec un code **422** et un message clair —
par exemple un âge négatif ou un genre inconnu. Teste-le toi-même en
envoyant une valeur absurde via `/docs`.

## 5. Tester avec curl (optionnel, en ligne de commande)

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"gender\":\"Male\",\"age\":67,\"hypertension\":0,\"heart_disease\":1,\"ever_married\":\"Yes\",\"work_type\":\"Private\",\"Residence_type\":\"Urban\",\"avg_glucose_level\":228.69,\"bmi\":36.6,\"smoking_status\":\"formerly smoked\"}"
```

## ✅ Checklist de fin d'étape 5 (partie API)

- [ ] `model_final.joblib` et `model_metadata.json` générés
- [ ] API lancée en local sans erreur
- [ ] `/docs` accessible et testable
- [ ] Une prédiction cohérente obtenue (patient à risque vs patient jeune/sain)
- [ ] Validation des erreurs testée (donnée invalide → 422)

## ➡️ Prochaine étape
Une fois l'API validée en local, on passera à :
1. **Commit + push** du nouveau code (`api.py`, `save_final_model.py`) sur GitHub
2. **Déploiement cloud** (on choisira ensemble la plateforme)
3. **CI/CD avec GitHub Actions**
