# Étape 3 — Définition du score métier

## Pourquoi l'accuracy ne suffit PAS ici

Avec ~95% de patients sans AVC dans le dataset, un modèle qui prédit
**toujours "pas d'AVC"** obtiendrait déjà ~95% d'accuracy... tout en étant
totalement inutile (il ne détecterait jamais aucun cas réel). C'est
l'exemple classique qui justifie de construire un score métier adapté.

## Le raisonnement métier retenu

| Cas | Situation | Conséquence | Coût retenu |
|---|---|---|---|
| **Faux Négatif (FN)** | Le modèle dit "pas de risque" à un patient qui va faire un AVC | Absence de prévention, séquelles graves possibles, urgence non anticipée | **+500** |
| **Faux Positif (FP)** | Le modèle alerte à tort | Examens complémentaires inutiles, stress | **+50** |
| **Vrai Positif (TP)** | Détection correcte d'un risque réel | Prise en charge préventive possible | **-100** (bénéfice) |
| **Vrai Négatif (TN)** | Rien à signaler, correctement | Situation normale | **0** |

**Ratio FN/FP = 10** : on considère qu'un AVC manqué coûte 10 fois plus
cher qu'une fausse alerte. C'est un choix de départ raisonnable et
défendable ; à toi de le justifier dans ton rapport (tu peux t'appuyer
sur le fait qu'un AVC non traité a des conséquences potentiellement
irréversibles et coûteuses à long terme, contre un simple examen de
dépistage supplémentaire).

## Ce que fait le script `business_score.py`

### 1. `business_cost(y_true, y_pred)`
Calcule le coût total en unités arbitraires à partir de la matrice de
confusion. **Plus bas = meilleur.**

### 2. `business_score(y_true, y_pred)`
Normalise ce coût en un score entre 0 et 1 (0 = pire cas possible,
c'est-à-dire ne jamais rien détecter ; 1 = détection parfaite).
**Plus haut = meilleur.** Cette version est utilisable directement dans
`GridSearchCV` via `business_scorer` (déjà prêt dans le script).

### 3. `find_optimal_threshold(y_true, y_proba)`
**Point clé** : par défaut, un modèle de classification utilise un seuil
de 0.5 pour décider "positif" ou "négatif". Mais vu notre matrice de
coûts (FN bien plus cher que FP), il est presque toujours préférable
d'**abaisser ce seuil** pour détecter plus de cas à risque, quitte à
avoir plus de fausses alertes. Cette fonction teste plusieurs seuils et
trouve celui qui minimise le coût métier total.

## Comment lancer le test

```bash
python src\business_score.py
```

Tu verras s'afficher :
- le coût avec le seuil par défaut (0.5)
- le seuil optimal trouvé
- le coût avec ce seuil optimal
- le gain obtenu

## Comment l'utiliser à l'Étape 4

Dans ton script d'entraînement de modèles, importe ces fonctions :

```python
from business_score import business_scorer, business_cost, find_optimal_threshold

# Dans un GridSearchCV, remplace scoring='accuracy' par :
grid_search = GridSearchCV(model, param_grid, scoring=business_scorer, cv=5)
```

Cela permettra de sélectionner les hyperparamètres qui minimisent
**réellement** le coût métier, et non une métrique générique déconnectée
de l'enjeu.

## ✅ Checklist de fin d'étape 3

- [ ] Matrice de coûts définie et justifiée dans ton rapport
- [ ] Script `business_score.py` testé et fonctionnel
- [ ] Compris le principe du seuil optimal (threshold tuning)
- [ ] Prêt à intégrer `business_scorer` dans le GridSearchCV de l'Étape 4

## ➡️ Prochaine étape
Étape 4 : Entraînement et comparaison de plusieurs modèles (régression
logistique, random forest, XGBoost), avec GridSearchCV optimisé sur ce
score métier, gestion du déséquilibre (SMOTE / pondération), et analyse
de la feature importance (SHAP/LIME).
