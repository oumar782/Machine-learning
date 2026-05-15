# api_avec_date.py - API avec logique avancée de sélection par date
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
import io
import uvicorn
import warnings
warnings.filterwarnings('ignore')

# =========================
# INITIALISATION
# =========================
app = FastAPI(
    title="Football Pitch - Prédiction d'Affluence ULTRA",
    description="Prédiction intelligente basée sur la date (jour, saison, vacances)",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODÈLES PYDANTIC
# =========================
class PredictionRequest(BaseModel):
    date: str  # Format: "2026-05-20"
    terrain: Optional[str] = "tous"

class DateAnalysisResponse(BaseModel):
    date: str
    jour_semaine: str
    type_jour: str  # "Semaine", "Weekend", "Férié", "Vacances"
    saison: str
    coefficient_affluence: float
    recommandation_generale: str

# =========================
# LOGIQUE AVANCÉE DE SÉLECTION PAR DATE
# =========================

# Jours fériés France 2025-2026 (adaptez selon votre pays)
JOURS_FERIES = [
    "2025-01-01", "2025-04-21", "2025-05-01", "2025-05-08", "2025-05-29",
    "2025-06-09", "2025-07-14", "2025-08-15", "2025-11-01", "2025-11-11",
    "2025-12-25", "2026-01-01", "2026-04-06", "2026-05-01", "2026-05-14",
    "2026-05-25", "2026-07-14", "2026-08-15", "2026-11-01", "2026-11-11",
    "2026-12-25"
]

# Vacances scolaires (simplifié - zone C)
VACANCES = {
    'printemps': [("2025-04-12", "2025-04-28"), ("2026-04-04", "2026-04-20")],
    'ete': [("2025-07-05", "2025-09-01"), ("2026-07-04", "2026-09-01")],
    'toussaint': [("2025-10-18", "2025-11-03"), ("2026-10-17", "2026-11-02")],
    'noel': [("2025-12-20", "2026-01-05"), ("2026-12-19", "2027-01-04")],
    'hiver': [("2026-02-14", "2026-03-02")]
}

def analyser_date(date_str):
    """Analyse intelligente de la date pour ajuster les prédictions"""
    
    date_obj = pd.to_datetime(date_str)
    jour_semaine_num = date_obj.dayofweek
    jours_noms = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    jour_nom = jours_noms[jour_semaine_num]
    
    # 1. Type de jour
    if date_str in JOURS_FERIES:
        type_jour = "Férié"
        coefficient_base = 1.5
    elif jour_semaine_num >= 5:  # Samedi (5) ou Dimanche (6)
        type_jour = "Weekend"
        coefficient_base = 1.3
    else:
        type_jour = "Semaine"
        coefficient_base = 0.8
    
    # 2. Saison
    mois = date_obj.month
    if mois in [12, 1, 2]:
        saison = "Hiver"
        coefficient_saison = 0.7
    elif mois in [3, 4, 5]:
        saison = "Printemps"
        coefficient_saison = 1.0
    elif mois in [6, 7, 8]:
        saison = "Été"
        coefficient_saison = 1.2
    else:
        saison = "Automne"
        coefficient_saison = 0.9
    
    # 3. Vacances scolaires
    est_vacances = False
    for periode, dates in VACANCES.items():
        for start, end in dates:
            if start <= date_str <= end:
                est_vacances = True
                coefficient_vacances = 1.2
                break
    
    if est_vacances:
        type_jour = f"Vacances ({type_jour})"
        coefficient_vacances = 1.2
    else:
        coefficient_vacances = 1.0
    
    # 4. Heures spécifiques selon le type de jour
    if type_jour in ["Weekend", "Férié"]:
        heures_pointe = [14, 15, 16, 17, 18, 19, 20, 21, 22]
        heures_creuses = [8, 9, 10, 11]
    else:  # Semaine
        heures_pointe = [18, 19, 20, 21, 22]
        heures_creuses = [9, 10, 11, 12, 13, 14]
    
    # Coefficient final
    coefficient_final = coefficient_base * coefficient_saison * coefficient_vacances
    coefficient_final = round(coefficient_final, 2)
    
    # Recommandation
    if coefficient_final >= 1.5:
        recommandation = "🔴 ATTENTION: Très forte affluence attendue - Réservez à l'avance!"
    elif coefficient_final >= 1.2:
        recommandation = "🟠 Affluence élevée - Pensez à réserver"
    elif coefficient_final >= 0.9:
        recommandation = "🟡 Affluence normale"
    else:
        recommandation = "🟢 Affluence calme - Bon moment pour jouer"
    
    return {
        'date': date_str,
        'date_obj': date_obj,
        'jour_semaine_num': jour_semaine_num,
        'jour_nom': jour_nom,
        'type_jour': type_jour,
        'saison': saison,
        'est_vacances': est_vacances,
        'coefficient_affluence': coefficient_final,
        'heures_pointe': heures_pointe,
        'heures_creuses': heures_creuses,
        'recommandation': recommandation,
        'mois': mois,
        'jour_mois': date_obj.day
    }

# =========================
# VARIABLES GLOBALES
# =========================
df = None
model = None
le = None
feature_names = None
score = 0
jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
terrains_liste = []

# =========================
# CHARGEMENT DES DONNÉES
# =========================
def load_and_train():
    global df, model, le, feature_names, score, terrains_liste
    
    print("\n" + "="*60)
    print("🏆 CHARGEMENT DES DONNÉES RÉELLES")
    print("="*60)
    
    url = "postgresql://postgres.yhnimydmntjucstxxlrx:oumar%40196678@aws-1-us-west-1.pooler.supabase.com:6543/postgres"
    engine = create_engine(url)
    
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("✅ Connexion à Supabase établie")
        
        df = pd.read_sql("SELECT * FROM reservation", engine)
        print(f"✅ {len(df)} réservations chargées")
        
    except Exception as e:
        print(f"⚠️ Erreur: {e}, création de données démo...")
        
        # Données de démonstration avec logique de date
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2025-12-31', freq='D')
        terrains = ['Terrain 1 (Foot5)', 'Terrain 2 (Foot7)', 'Terrain 3 (Foot5)', 
                    'Terrain 4 (Foot7)', 'Terrain 5 (Foot11)']
        
        reservations = []
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            analyse = analyser_date(date_str)
            
            # Nombre de réservations selon la date
            nbr_reservations = int(20 * analyse['coefficient_affluence'])
            
            for _ in range(nbr_reservations):
                # Heure selon le type de jour
                if analyse['type_jour'] in ["Weekend", "Férié"]:
                    heure = np.random.choice([14, 15, 16, 17, 18, 19, 20, 21], 
                                            p=[0.05, 0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1])
                else:
                    heure = np.random.choice([18, 19, 20, 21, 22], 
                                            p=[0.2, 0.25, 0.3, 0.15, 0.1])
                
                terrain = np.random.choice(terrains, p=[0.2, 0.3, 0.15, 0.25, 0.1])
                
                reservations.append({
                    'datereservation': date,
                    'heurereservation': f"{heure:02d}:00:00",
                    'nomterrain': terrain
                })
        
        df = pd.DataFrame(reservations)
        print(f"✅ {len(df)} réservations de démonstration créées")
    
    # Préparation
    df['datereservation'] = pd.to_datetime(df['datereservation'])
    df['heurereservation'] = pd.to_datetime(df['heurereservation'], format='%H:%M:%S', errors='coerce').dt.hour
    df = df.dropna(subset=['heurereservation'])
    df['heurereservation'] = df['heurereservation'].astype(int)
    
    df['jour_semaine'] = df['datereservation'].dt.dayofweek
    df['jour_mois'] = df['datereservation'].dt.day
    df['mois'] = df['datereservation'].dt.month
    df['nomterrain'] = df['nomterrain'].astype(str)
    
    terrains_liste = df['nomterrain'].unique().tolist()
    
    # Modèle
    le = LabelEncoder()
    df['terrain_code'] = le.fit_transform(df['nomterrain'])
    
    affluence = df.groupby([
        'jour_semaine', 'jour_mois', 'mois', 'heurereservation', 'nomterrain', 'terrain_code'
    ]).size().reset_index(name='nb_reservations')
    
    feature_names = ['jour_semaine', 'jour_mois', 'mois', 'heurereservation', 'terrain_code']
    X = affluence[feature_names]
    y = affluence['nb_reservations']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    print(f"\n✅ MODÈLE ENTRAÎNÉ - Score: {score*100:.1f}%")
    print(f"   Terrains: {terrains_liste}")
    print("="*60 + "\n")

# =========================
# FONCTION DE PRÉDICTION AVEC ANALYSE DE DATE
# =========================
def predire_affluence_avec_date(date_input, terrain_input="tous"):
    """Prédiction intégrant l'analyse intelligente de la date"""
    
    # Analyser la date
    analyse = analyser_date(date_input)
    
    date_obj = analyse['date_obj']
    jour_semaine = analyse['jour_semaine_num']
    jour_mois = analyse['jour_mois']
    mois = analyse['mois']
    jour_nom = analyse['jour_nom']
    
    # Obtenir les terrains
    if terrain_input == "tous":
        terrains = terrains_liste
    else:
        terrains = [t for t in terrains_liste if terrain_input.lower() in t.lower()]
        if not terrains:
            raise ValueError(f"Terrain '{terrain_input}' non trouvé")
    
    resultats = []
    
    for terrain in terrains:
        terrain_code = le.transform([terrain])[0]
        predictions_par_heure = []
        
        for heure in range(0, 24):
            exemple = pd.DataFrame([[jour_semaine, jour_mois, mois, heure, terrain_code]], 
                                  columns=feature_names)
            prediction = model.predict(exemple)[0]
            
            # Appliquer le coefficient de la date
            prediction = prediction * analyse['coefficient_affluence']
            pred_int = int(round(prediction))
            predictions_par_heure.append(pred_int)
        
        max_pred = max(predictions_par_heure)
        heure_pointe = predictions_par_heure.index(max_pred)
        
        # Ajuster l'heure de pointe selon l'analyse
        if heure_pointe not in analyse['heures_pointe'] and analyse['heures_pointe']:
            heure_pointe = analyse['heures_pointe'][0]
        
        resultats.append({
            'terrain': terrain,
            'predictions': predictions_par_heure,
            'max_prediction': max_pred,
            'heure_pointe': heure_pointe,
            'total_journee': sum(predictions_par_heure)
        })
    
    resultats.sort(key=lambda x: x['max_prediction'], reverse=True)
    top = resultats[0]
    
    # Créer la prédiction texte
    heure_debut = max(8, top['heure_pointe'] - 2)
    heure_fin = min(23, top['heure_pointe'] + 2)
    
    prediction_texte = f"Ce {jour_nom} entre {heure_debut}h et {heure_fin}h, le {top['terrain']} risque d'être complet!"
    
    return {
        'analyse_date': analyse,
        'date': date_input,
        'jour': jour_nom,
        'type_jour': analyse['type_jour'],
        'saison': analyse['saison'],
        'coefficient': analyse['coefficient_affluence'],
        'heure_critique': f"{heure_debut}h - {heure_fin}h",
        'terrain_critique': top['terrain'],
        'prediction_texte': prediction_texte,
        'affluence_score': top['max_prediction'],
        'recommandation': analyse['recommandation'],
        'terrains': resultats,
        'score_modele': round(score, 3)
    }

# =========================
# ENDPOINTS API
# =========================

@app.on_event("startup")
async def startup():
    load_and_train()

@app.get("/")
def root():
    return {
        "message": "🏆 API Prédiction d'Affluence avec Logique de Date",
        "version": "5.0.0",
        "fonctionnalites": {
            "analyse_date": "✅ Prend en compte jour, saison, vacances, jours fériés",
            "coefficient_affluence": "✅ Ajuste les prédictions selon la date",
            "heures_pointe": "✅ Différencie weekend/semaine"
        },
        "endpoints": {
            "/": "GET - Cette page",
            "/analyze-date": "POST - Analyser une date uniquement",
            "/patterns": "GET - Patterns appris",
            "/predict/simple": "POST - Prédiction simple",
            "/predict/full": "POST - Prédiction complète",
            "/predict/compare": "POST - Comparer deux dates",
            "/docs": "GET - Documentation"
        }
    }

@app.post("/analyze-date", response_model=DateAnalysisResponse)
async def analyze_date(request: PredictionRequest):
    """Analyse uniquement la date (sans prédiction terrain)"""
    try:
        analyse = analyser_date(request.date)
        return DateAnalysisResponse(
            date=analyse['date'],
            jour_semaine=analyse['jour_nom'],
            type_jour=analyse['type_jour'],
            saison=analyse['saison'],
            coefficient_affluence=analyse['coefficient_affluence'],
            recommandation_generale=analyse['recommandation']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patterns")
def get_patterns():
    """Retourne les patterns appris par le modèle"""
    return {
        "message": "Patterns basés sur l'historique des réservations",
        "model_score": f"{score*100:.1f}%",
        "terrains_disponibles": terrains_liste
    }

@app.post("/predict/simple")
async def predict_simple(request: PredictionRequest):
    """Prédiction simple avec analyse de date intégrée"""
    try:
        result = predire_affluence_avec_date(request.date, request.terrain)
        
        return {
            "date": request.date,
            "jour": result['jour'],
            "type_jour": result['type_jour'],
            "coefficient_affluence": result['coefficient'],
            "prediction": result['prediction_texte'],
            "recommandation": result['recommandation'],
            "score_modele": result['score_modele']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/full")
async def predict_full(request: PredictionRequest):
    """Prédiction complète avec tous les détails de l'analyse de date"""
    try:
        result = predire_affluence_avec_date(request.date, request.terrain)
        
        # Générer les heures recommandées
        heures_recommandees = []
        for heure in result['analyse_date']['heures_creuses'][:3]:
            heures_recommandees.append(f"{heure}h-{heure+1}h")
        
        return {
            "date_analyse": {
                "date": result['date'],
                "jour": result['jour'],
                "type": result['type_jour'],
                "saison": result['saison'],
                "coefficient_affluence": result['coefficient']
            },
            "prediction": {
                "texte": result['prediction_texte'],
                "terrain_risque": result['terrain_critique'],
                "heure_critique": result['heure_critique'],
                "affluence_score": result['affluence_score']
            },
            "recommandations": {
                "generale": result['recommandation'],
                "heures_calmes": heures_recommandees,
                "conseil": "Réservez à l'avance si vous voulez jouer en heure de pointe"
            },
            "score_modele": result['score_modele']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/compare")
async def compare_dates(request: PredictionRequest, date2: str):
    """Compare deux dates différentes"""
    try:
        result1 = predire_affluence_avec_date(request.date, request.terrain)
        result2 = predire_affluence_avec_date(date2, request.terrain)
        
        # Déterminer quelle date est la plus chargée
        if result1['coefficient'] > result2['coefficient']:
            plus_charge = request.date
            difference = result1['coefficient'] - result2['coefficient']
        else:
            plus_charge = date2
            difference = result2['coefficient'] - result1['coefficient']
        
        return {
            "date1": {
                "date": request.date,
                "jour": result1['jour'],
                "type": result1['type_jour'],
                "coefficient": result1['coefficient'],
                "prediction": result1['prediction_texte']
            },
            "date2": {
                "date": date2,
                "jour": result2['jour'],
                "type": result2['type_jour'],
                "coefficient": result2['coefficient'],
                "prediction": result2['prediction_texte']
            },
            "comparaison": {
                "date_la_plus_chargee": plus_charge,
                "difference_coefficient": round(difference, 2),
                "conseil": f"Privilégiez {plus_charge if plus_charge == request.date else date2} pour éviter la foule"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/calendar")
async def get_month_calendar(start_date: str, end_date: str, terrain: str = "tous"):
    """Génère un calendrier des prédictions sur une période"""
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        dates = pd.date_range(start=start, end=end, freq='D')
        calendar = []
        
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            analyse = analyser_date(date_str)
            
            calendar.append({
                "date": date_str,
                "jour": analyse['jour_nom'],
                "type": analyse['type_jour'],
                "coefficient": analyse['coefficient_affluence'],
                "recommandation": analyse['recommandation']
            })
        
        # Statistiques sur la période
        coefficients = [c['coefficient'] for c in calendar]
        
        return {
            "periode": f"{start_date} au {end_date}",
            "nbr_jours": len(calendar),
            "moyenne_affluence": round(sum(coefficients) / len(coefficients), 2),
            "jours_critiques": [c for c in calendar if c['coefficient'] >= 1.5],
            "jours_calmes": [c for c in calendar if c['coefficient'] <= 0.8],
            "calendrier": calendar
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# LANCEMENT
# =========================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔮 PRÉDICTION AVEC LOGIQUE DE SÉLECTION PAR DATE")
    print("="*60)
    
    print("\n📌 EXEMPLES DE REQUÊTES:")
    print("   1. Analyser une date: POST /analyze-date")
    print('      Body: {"date": "2026-05-20", "terrain": "tous"}')
    print("\n   2. Prédiction simple: POST /predict/simple")
    print('      Body: {"date": "2026-05-20", "terrain": "tous"}')
    print("\n   3. Comparer deux dates: POST /predict/compare?date2=2026-07-14")
    print('      Body: {"date": "2026-05-20", "terrain": "tous"}')
    print("\n   4. Calendrier mensuel: POST /predict/calendar")
    print('      Body: {"start_date": "2026-05-01", "end_date": "2026-05-31", "terrain": "tous"}')
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")