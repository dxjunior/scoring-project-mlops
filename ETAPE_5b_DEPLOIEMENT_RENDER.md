# Étape 5 (suite) — Déploiement sur Render

## 1. Committe et pousse ton code sur GitHub

```bash
git add .
git status
```

**Vérifie bien** que `model_final.joblib` et `model_metadata.json`
apparaissent dans la liste (contrairement aux données et aux artefacts
MLFlow, ces deux fichiers **doivent** être versionnés : Render en a
besoin pour faire fonctionner l'API en ligne).

```bash
git commit -m "Add FastAPI scoring API + final trained model"
git push
```

## 2. Crée un compte Render

Va sur **https://render.com** et inscris-toi (tu peux te connecter
directement avec ton compte GitHub, c'est plus rapide).

## 3. Crée un nouveau "Web Service"

1. Dans le dashboard Render, clique sur **"New"** → **"Web Service"**
2. Autorise Render à accéder à ton compte GitHub si demandé
3. Sélectionne ton dépôt **`scoring-project-mlops`**

## 4. Configure le service

| Champ | Valeur |
|---|---|
| **Name** | `scoring-api-avc` (ou ce que tu veux) |
| **Region** | Frankfurt (le plus proche de toi) |
| **Branch** | `main` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn src.api:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | **Free** |

⚠️ Le `$PORT` est important : Render assigne dynamiquement un port,
il ne faut **pas** mettre `--port 8000` en dur ici.

## 5. Clique sur "Create Web Service"

Render va :
1. Cloner ton dépôt
2. Exécuter le build command (installer les dépendances — patience,
   ça peut prendre 3-5 minutes la première fois)
3. Lancer le start command

Tu peux suivre les logs en direct dans le dashboard. Quand tu vois
`Application startup complete` dans les logs, c'est bon.

## 6. Teste ton API en ligne

Render te donne une URL du type :
```
https://scoring-api-avc.onrender.com
```

Va sur `https://scoring-api-avc.onrender.com/docs` — tu dois retrouver
exactement la même interface Swagger que celle testée en local !

## ⚠️ Limitation du plan gratuit à connaître (à mentionner dans ton rapport)

- L'API **s'endort après 15 minutes d'inactivité**
- La première requête après une période d'inactivité prend **30 à 60
  secondes** (le temps que Render "réveille" le service) — c'est normal,
  ce n'est pas un bug
- Les requêtes suivantes sont rapides tant que le service reste actif

C'est un bon point à discuter dans ta soutenance : en production réelle,
on choisirait un plan payant pour éviter ce cold start.

## ✅ Checklist de fin de déploiement

- [ ] Code poussé sur GitHub avec `model_final.joblib` et `model_metadata.json` inclus
- [ ] Service Render créé et déployé sans erreur
- [ ] `/docs` accessible publiquement via l'URL Render
- [ ] Une prédiction testée avec succès en ligne (pas juste en local)

## ➡️ Prochaine étape
CI/CD avec GitHub Actions : on va automatiser le déploiement pour que
chaque `git push` redéploie automatiquement l'API (bonus : Render le
fait déjà par défaut à chaque push sur `main` ! On ajoutera surtout des
**tests automatiques** avant déploiement).
