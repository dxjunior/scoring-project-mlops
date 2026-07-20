"""
Étape 3 - Définition du score métier
======================================
Contexte : prédiction de risque d'AVC (stroke).

On définit une matrice de coûts métier pour chaque type d'erreur, puis
une fonction de score qui permet de comparer les modèles selon un
critère aligné sur l'enjeu réel (et non seulement l'accuracy classique,
inadaptée ici vu le déséquilibre des classes ~95%/5%).

RAISONNEMENT MÉTIER
-------------------
Dans un contexte médical de dépistage de risque d'AVC :

- Faux Négatif (FN) : le modèle dit "pas de risque" à un patient qui va
  réellement faire un AVC.
  -> Conséquence : absence de prise en charge préventive, risque de
     séquelles graves, invalidité, voire décès. Coût humain énorme,
     coût financier du traitement d'urgence + rééducation potentiels.
  -> COÛT ARBITRAIRE RETENU : 500 (unité conventionnelle)

- Faux Positif (FP) : le modèle alerte un patient qui n'aura pas d'AVC.
  -> Conséquence : examens complémentaires (IRM, consultation
     neurologue), stress pour le patient, coût du système de santé.
  -> COÛT ARBITRAIRE RETENU : 50

- Vrai Positif (TP) : le modèle détecte correctement un risque réel.
  -> Bénéfice : prise en charge préventive possible, patient
     potentiellement sauvé de complications.
  -> "COÛT" RETENU : -100 (un gain, donc coût négatif)

- Vrai Négatif (TN) : le modèle ne détecte rien, et c'est correct.
  -> Aucun coût ni bénéfice particulier.
  -> COÛT RETENU : 0

Ratio FN/FP = 10 : on considère qu'un AVC manqué coûte 10 fois plus
cher qu'une fausse alerte. Ce ratio est un point de départ ; il devrait
idéalement être ajusté avec un expert médical / des données de coûts
réels de santé publique pour un vrai déploiement.

⚠️ IMPORTANT POUR TON RAPPORT : ces valeurs sont volontairement
illustratives et rondes. Dans ton document de soutenance, tu dois
justifier/discuter ce choix (tu peux citer des ordres de grandeur du
coût réel d'une prise en charge d'AVC vs. un examen de dépistage,
ou simplement assumer que c'est une hypothèse de travail raisonnable
à documenter comme telle).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, make_scorer


# -----------------------------------------------------------------
# 1. Matrice de coûts métier
# -----------------------------------------------------------------
COST_FN = 500   # Faux négatif : AVC manqué
COST_FP = 50    # Faux positif : fausse alerte
COST_TP = -100  # Vrai positif : détection réussie (bénéfice)
COST_TN = 0     # Vrai négatif : rien à signaler, normal


def business_cost(y_true, y_pred):
    """
    Calcule le coût métier total pour un jeu de prédictions.
    Coût total plus BAS = meilleur modèle.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (
        fn * COST_FN
        + fp * COST_FP
        + tp * COST_TP
        + tn * COST_TN
    )
    return total_cost


def business_score(y_true, y_pred):
    """
    Version normalisée du coût métier, transformée en score entre 0 et 1
    (plus HAUT = meilleur), pour être comparable à d'autres métriques
    (accuracy, f1, etc.) et utilisable dans GridSearchCV.

    On compare le coût du modèle au pire cas possible (tout prédire
    en faux négatif, i.e. ne jamais rien détecter) pour obtenir un
    pourcentage d'amélioration.
    """
    y_true = np.asarray(y_true)
    n_positives = y_true.sum()
    n_negatives = len(y_true) - n_positives

    # Pire cas de référence : le modèle ne détecte jamais aucun stroke
    worst_case_cost = n_positives * COST_FN + n_negatives * COST_TN

    # Meilleur cas théorique : détection parfaite
    best_case_cost = n_positives * COST_TP + n_negatives * COST_TN

    actual_cost = business_cost(y_true, y_pred)

    # Score = position entre le pire cas (0) et le meilleur cas (1)
    denom = (worst_case_cost - best_case_cost)
    if denom == 0:
        return 1.0
    score = (worst_case_cost - actual_cost) / denom
    return score


# Scorer utilisable directement dans GridSearchCV / cross_val_score
business_scorer = make_scorer(business_score, greater_is_better=True)


def find_optimal_threshold(y_true, y_proba, thresholds=None):
    """
    Cherche le seuil de décision (au lieu du seuil par défaut 0.5) qui
    MINIMISE le coût métier total. C'est une étape essentielle : en
    médical, un seuil différent de 0.5 est presque toujours préférable
    pour privilégier le rappel (recall) sur la classe à risque.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.01)

    best_threshold = 0.5
    best_cost = float("inf")
    results = []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        cost = business_cost(y_true, y_pred_t)
        results.append({"threshold": round(t, 2), "cost": cost})
        if cost < best_cost:
            best_cost = cost
            best_threshold = t

    results_df = pd.DataFrame(results)
    return best_threshold, best_cost, results_df


# -----------------------------------------------------------------
# 2. Démonstration / test
# -----------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier

    train = pd.read_csv("data/train_processed.csv")
    test = pd.read_csv("data/test_processed.csv")

    X_train = train.drop(columns=["stroke"])
    y_train = train["stroke"]
    X_test = test.drop(columns=["stroke"])
    y_test = test["stroke"]

    model = RandomForestClassifier(random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred_default = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print("ÉVALUATION AVEC SEUIL PAR DÉFAUT (0.5)")
    print("=" * 60)
    cost_default = business_cost(y_test, y_pred_default)
    score_default = business_score(y_test, y_pred_default)
    print(f"Coût métier total   : {cost_default}")
    print(f"Score métier (0-1)  : {score_default:.4f}")

    print("\n" + "=" * 60)
    print("RECHERCHE DU SEUIL OPTIMAL (minimisation du coût métier)")
    print("=" * 60)
    best_t, best_cost, results_df = find_optimal_threshold(y_test, y_proba)
    print(f"Seuil optimal       : {best_t:.2f}")
    print(f"Coût métier minimal : {best_cost}")

    y_pred_optimal = (y_proba >= best_t).astype(int)
    score_optimal = business_score(y_test, y_pred_optimal)
    print(f"Score métier optimal (0-1) : {score_optimal:.4f}")

    print(f"\n💡 Gain en changeant le seuil : {cost_default - best_cost} unités de coût économisées")
