# MLengine

Ce projet propose une API FastAPI et une interface Streamlit pour la prédiction d'affluence de terrains.

## Exécution locale

1. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Lancer l'API :
   ```bash
   python app.py
   ```
3. Lancer l'interface Streamlit :
   ```bash
   streamlit run streamlit_app.py
   ```

## Déploiement

- Streamlit Cloud : déployer `streamlit_app.py` comme application Streamlit.
- Vercel : le fichier `vercel.json` utilise `app.py` comme point d'entrée API.

## Structure

- `app.py` : API FastAPI + modèle
- `streamlit_app.py` : interface utilisateur Streamlit
- `requirements.txt` : dépendances Python
- `Procfile` : déploiement Heroku / Streamlit compatible
- `vercel.json` : déploiement Vercel pour l'API
