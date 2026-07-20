# Étape 4 — Entraînement et comparaison de modèles

## ⚠️ Avant de lancer : temps d'exécution
Ce script fait du GridSearchCV sur 3 modèles avec SMOTE — sur ton vrai
dataset (5110 lignes), ça peut prendre **quelques minutes**. C'est
normal, laisse tourner.

## Ce que fait le script

### 1. Baseline
Un `DummyClassifier` qui prédit toujours la classe majoritaire (pas
d'AVC). Sert de référence minimale : tout modèle sérieux doit faire
mieux que ça sur le score métier.

### 2. Trois modèles comparés
- **Logistic Regression** (simple, interprétable, bonne baseline "sérieuse")
- **Random Forest** (non-linéaire, gère bien les interactions)
- **XGBoost** (souvent le plus performant sur données tabulaires)

Chacun est optimisé via `GridSearchCV` avec **`scoring=business_scorer`**
(notre score métier de l'Étape 3) — **pas** l'accuracy. C'est un point
important à souligner dans ton rapport : on optimise directement ce
qui compte pour le métier, pas une métrique générique.

### 3. Gestion du déséquilibre : SMOTE
Chaque modèle est intégré dans un pipeline `imblearn` qui applique
**SMOTE** (Synthetic Minority Oversampling) uniquement sur les données
d'entraînement, à chaque fold de la cross-validation. Ça génère des
exemples synthétiques de la classe minoritaire (stroke=1) pour
rééquilibrer l'apprentissage, sans jamais toucher au jeu de test.

### 4. Seuil optimal appliqué systématiquement
Pour chaque modèle, en plus des métriques au seuil par défaut (0.5),
le script cherche le seuil qui minimise le coût métier (comme à
l'Étape 3) et évalue le modèle avec ce seuil optimal.

### 5. Feature importance
- **Importance native** : pour Random Forest et XGBoost, importance
  basée sur la réduction d'impureté (Gini) — rapide mais parfois biaisée
  vers les variables à forte cardinalité.
- **SHAP** : plus robuste et interprétable, montre la contribution
  moyenne de chaque feature à la prédiction. C'est la référence
  actuelle en interprétabilité de modèles (à mentionner explicitement
  dans ton rapport comme alternative à LIME, mentionné dans le sujet).

### 6. Tout est journalisé dans MLFlow
Chaque modèle = une run MLFlow avec :
- paramètres (meilleurs hyperparamètres trouvés par GridSearchCV)
- métriques (accuracy, f1, recall, precision, roc_auc, coût métier,
  score métier, seuil optimal)
- artefacts (matrice de confusion, feature importance, SHAP, modèle
  sauvegardé)

## Comment lancer

```bash
python src\train_models.py
```

Patiente (potentiellement plusieurs minutes sur les vraies données).
À la fin, tu verras un tableau comparatif dans le terminal, et un
fichier `model_comparison.csv` sera créé.

## Explorer les résultats

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Va dans l'expérience **projet_scoring_modeles**, compare les runs entre
elles (MLFlow permet de sélectionner plusieurs runs et de comparer leurs
métriques côte à côte).

## Points à documenter dans ton rapport (important pour la soutenance)

1. **Pourquoi SMOTE et pas juste `class_weight='balanced'`** ? Tu peux
   tester les deux et comparer — SMOTE génère des données synthétiques,
   `class_weight` pénalise juste les erreurs sur la classe minoritaire
   sans créer de nouvelles données. Les deux sont valables, à toi de
   choisir/comparer.
2. **Le modèle gagnant selon le score métier est-il le même que celui
   gagnant selon l'accuracy ou le ROC AUC ?** Souvent non — c'est
   précisément l'intérêt de la démarche.
3. **Analyse des features importantes** : est-ce que les variables
   ressortant en tête (âge, hypertension, glucose...) sont cohérentes
   avec les connaissances médicales sur les facteurs de risque d'AVC ?
   Ça renforce la crédibilité de ton modèle.

## ✅ Checklist de fin d'étape 4

- [ ] Script exécuté sans erreur sur les vraies données
- [ ] `model_comparison.csv` généré
- [ ] Runs visibles et comparables dans l'interface MLFlow
- [ ] Un modèle "gagnant" identifié selon le score métier
- [ ] Feature importance / SHAP analysés et cohérents médicalement

## ➡️ Prochaine étape
Étape 5 : Déploiement de l'API de scoring avec FastAPI, versioning Git,
et pipeline CI/CD avec GitHub Actions.
