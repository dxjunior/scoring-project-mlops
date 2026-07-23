# Étape 5 (complément) — CI/CD avec GitHub Actions

## Le principe

Jusqu'ici, Render redéployait automatiquement à chaque `git push`, **sans
aucune vérification**. Un vrai pipeline CI/CD ajoute une étape de
**contrôle qualité obligatoire** : les tests doivent passer avant que
le déploiement ne soit autorisé.

```
git push → GitHub Actions lance les tests
              ├── ✅ Tests OK  → déploiement déclenché sur Render
              └── ❌ Tests KO  → déploiement bloqué, tu es alerté
```

## 1. Place les nouveaux fichiers

```
scoring_project/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── tests/
│   └── test_api.py
└── ...
```

⚠️ Le dossier `.github` commence par un point — même piège que pour
`.gitignore` plus tôt. Vérifie avec `dir /a` qu'il s'appelle bien
`.github` et pas `github`.

## 2. Teste les tests en local d'abord

```bash
pip install pytest
pytest tests/ -v
```

Tu dois voir **11 tests**, la plupart doivent passer. Le test
`test_high_risk_profile_scores_higher_than_low_risk` vérifie que ton
modèle est médicalement cohérent (un profil à risque doit avoir une
probabilité plus élevée qu'un profil sain) — sur ton vrai modèle
entraîné sur les vraies données, il devrait passer. S'il échoue, c'est
un signal intéressant à creuser (et à mentionner dans ton rapport !).

## 3. Configure le "Deploy Hook" sur Render

Par défaut, Render redéploie automatiquement à chaque push — on va
**désactiver ça** et le remplacer par un déclenchement contrôlé par
GitHub Actions (donc conditionné aux tests).

1. Va sur ton dashboard Render → ton service `scoring-project-mlops`
2. Onglet **Settings**
3. Trouve **"Auto-Deploy"** → mets-le sur **"No"** (pour éviter le
   double déploiement)
4. Toujours dans Settings, trouve la section **"Deploy Hook"**
5. Copie l'URL du Deploy Hook (ressemble à
   `https://api.render.com/deploy/srv-xxxxx?key=yyyyy`)

## 4. Ajoute cette URL comme secret GitHub

1. Va sur ton dépôt GitHub → **Settings** (du dépôt, pas de ton profil)
2. **Secrets and variables** → **Actions**
3. **New repository secret**
4. Name : `RENDER_DEPLOY_HOOK_URL`
5. Value : colle l'URL copiée à l'étape précédente
6. **Add secret**

⚠️ Cette URL est un secret — ne la mets **jamais** en clair dans ton
code ou sur GitHub publiquement. C'est exactement pour ça qu'on utilise
les "Secrets" GitHub Actions.

## 5. Commit et push

```bash
git add .
git commit -m "Add CI/CD pipeline with GitHub Actions (tests + conditional deploy)"
git push
```

## 6. Observe le pipeline en action

Va sur ton dépôt GitHub → onglet **"Actions"**. Tu devrais voir ton
workflow **"CI/CD - Tests et déploiement"** en cours d'exécution, avec
deux jobs : `test` puis `deploy` (qui ne se lance que si `test` réussit).

Clique dessus pour voir les logs en détail, exactement comme dans ton
terminal local.

## Pourquoi c'est important pour ton rapport/soutenance

Ce mécanisme illustre un principe central du MLOps : **on ne déploie
jamais un modèle ou une API non testée en production**. Tu peux
présenter ce schéma à l'oral :

```
Développeur push du code
        ↓
GitHub Actions exécute automatiquement les tests
        ↓
   Tests passent ?
   ├── Oui → déploiement automatique sur Render
   └── Non → déploiement bloqué, le développeur est notifié par email/GitHub
```

## ✅ Checklist de fin de CI/CD

- [ ] `pytest tests/ -v` passe en local
- [ ] `.github/workflows/ci-cd.yml` et `tests/test_api.py` committés
- [ ] Deploy Hook Render configuré et Auto-Deploy désactivé
- [ ] Secret `RENDER_DEPLOY_HOOK_URL` ajouté sur GitHub
- [ ] Workflow visible et vert (✅) dans l'onglet "Actions" de GitHub
- [ ] Un déploiement réussi observé après un push

## ➡️ Prochaine étape
Étape 6 : Interface Streamlit pour permettre à un utilisateur non
technique de saisir des données patient et voir le résultat sans passer
par `/docs`.
