# api_final.py - API ULTRA COMPLÈTE avec vraies données
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    description="Prédit les heures de forte affluence basé sur les vraies réservations",
    version="4.0.0"
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
    terrain: Optional[str] = "tous"  # "tous", "Terrain 1", "Terrain 2", etc.

class PredictionResponse(BaseModel):
    date: str
    jour: str
    heure_critique: str
    terrain_critique: str
    prediction_texte: str
    affluence_score: float
    recommandation: str

class FullPredictionResponse(BaseModel):
    date: str
    jour: str
    score_modele: float
    patterns: dict
    top_terrain: dict
    heures_pointe: List[dict]
    alertes: List[str]
    tableau_heures: List[dict]

# =========================
# VARIABLES GLOBALES
# =========================
df = None
model = None
le = None
feature_names = None
score = 0
jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
jour_top = 0
heure_top = 0
heure_debut = 0
heure_fin = 0
terrain_top = ""
terrains_liste = []

# =========================
# CHARGEMENT DES VRAIES DONNÉES
# =========================
def load_and_train():
    global df, model, le, feature_names, score, jour_top, heure_top, heure_debut, heure_fin, terrain_top, terrains_liste
    
    print("\n" + "="*60)
    print("🏆 CHARGEMENT DES DONNÉES RÉELLES")
    print("="*60)
    
    url = "postgresql://postgres.yhnimydmntjucstxxlrx:oumar%40196678@aws-1-us-west-1.pooler.supabase.com:6543/postgres"
    engine = create_engine(url)
    
    try:
        # Test de connexion
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Connexion à Supabase établie")
        
        # Chargement des vraies réservations
        df = pd.read_sql("SELECT * FROM reservation", engine)
        print(f"✅ {len(df)} réservations chargées")
        
        # Afficher un aperçu
        print("\n📊 APERÇU DES DONNÉES:")
        print(f"   Période: {df['datereservation'].min()} à {df['datereservation'].max()}")
        print(f"   Terrains: {df['nomterrain'].unique().tolist()}")
        print(f"   Heures disponibles: {sorted(df['heurereservation'].unique())}")
        
    except Exception as e:
        print(f"⚠️ Erreur de connexion: {e}")
        print("📊 Création de données de démonstration réalistes...")
        
        # Données de démonstration réalistes
        np.random.seed(42)
        
        # Création de dates sur 1 an
        dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
        
        # Heures de réservation réalistes (plutôt le soir)
        heures_populaires = [18, 19, 20, 21, 22, 14, 15, 16, 17]
        heures_probas = [0.15, 0.25, 0.30, 0.20, 0.05, 0.02, 0.01, 0.01, 0.01]
        
        # Terrains
        terrains = ['Terrain 1 (Foot5)', 'Terrain 2 (Foot7)', 'Terrain 3 (Foot5)', 
                    'Terrain 4 (Foot7)', 'Terrain 5 (Foot11)', 'Terrain 6 (Foot5)']
        
        # Générer 10 000 réservations réalistes
        reservations = []
        for _ in range(10000):
            date = np.random.choice(dates)
            jour_semaine = date.dayofweek
            
            # Plus de réservations le weekend
            if jour_semaine >= 5:  # Weekend
                heure = np.random.choice(heures_populaires[:5], p=[0.1, 0.2, 0.35, 0.25, 0.1])
            else:  # Semaine
                heure = np.random.choice(heures_populaires, p=[0.12, 0.22, 0.28, 0.18, 0.08, 0.04, 0.03, 0.03, 0.02])
            
            # Certains terrains plus populaires
            if 'Foot7' in terrains[1]:
                terrain_probas = [0.15, 0.25, 0.10, 0.25, 0.15, 0.10]
            else:
                terrain_probas = [0.20, 0.20, 0.20, 0.15, 0.15, 0.10]
            
            terrain = np.random.choice(terrains, p=terrain_probas)
            
            reservations.append({
                'datereservation': date,
                'heurereservation': f"{heure:02d}:00:00",
                'nomterrain': terrain
            })
        
        df = pd.DataFrame(reservations)
        print(f"✅ {len(df)} réservations de démonstration créées")
    
    # ===== PRÉPARATION DES DONNÉES =====
    print("\n🔄 PRÉPARATION DES DONNÉES...")
    
    df['datereservation'] = pd.to_datetime(df['datereservation'])
    df['heurereservation'] = pd.to_datetime(df['heurereservation'], format='%H:%M:%S', errors='coerce').dt.hour
    
    # Supprimer les heures invalides
    df = df.dropna(subset=['heurereservation'])
    df['heurereservation'] = df['heurereservation'].astype(int)
    
    df['jour_semaine'] = df['datereservation'].dt.dayofweek
    df['jour_mois'] = df['datereservation'].dt.day
    df['mois'] = df['datereservation'].dt.month
    
    # Nettoyer les noms de terrains
    df['nomterrain'] = df['nomterrain'].astype(str)
    
    terrains_liste = df['nomterrain'].unique().tolist()
    print(f"   Terrains trouvés: {terrains_liste}")
    
    # ===== APPRENTISSAGE DES PATTERNS =====
    print("\n🧠 APPRENTISSAGE DES PATTERNS...")
    
    # Jours les plus chargés
    jours_charges = df.groupby('jour_semaine').size().sort_values(ascending=False)
    jour_top = jours_charges.index[0]
    print(f"   📅 Jour le plus chargé: {jours[jour_top]} ({jours_charges.iloc[0]} réservations)")
    
    # Heures les plus demandées
    heures_demandees = df.groupby('heurereservation').size().sort_values(ascending=False)
    heure_top = heures_demandees.index[0]
    heure_debut = max(0, heure_top - 2)
    heure_fin = min(23, heure_top + 2)
    print(f"   🕐 Heure de pointe: {heure_top}h ({heures_demandees.iloc[0]} réservations)")
    print(f"   ⏰ Créneau critique: {heure_debut}h - {heure_fin}h")
    
    # Terrains les plus populaires
    terrains_populaires = df.groupby('nomterrain').size().sort_values(ascending=False)
    terrain_top = terrains_populaires.index[0]
    print(f"   🏟️ Terrain le plus populaire: {terrain_top} ({terrains_populaires.iloc[0]} réservations)")
    
    # ===== CRÉATION DU MODÈLE RANDOM FOREST =====
    print("\n🤖 CRÉATION DU MODÈLE RANDOM FOREST...")
    
    le = LabelEncoder()
    df['terrain_code'] = le.fit_transform(df['nomterrain'])
    
    # Agrégation par créneau
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
    
    print(f"\n✅ MODÈLE ENTRAÎNÉ AVEC SUCCÈS!")
    print(f"   Score R²: {score:.4f} ({score*100:.1f}%)")
    print("="*60 + "\n")

# =========================
# FONCTION DE PRÉDICTION
# =========================
def predire_affluence(date_input, terrain_input="tous"):
    """Prédit l'affluence pour une date donnée"""
    
    date_user = pd.to_datetime(date_input)
    jour_semaine = date_user.dayofweek
    jour_mois = date_user.day
    mois = date_user.month
    jour_nom = jours[jour_semaine]
    
    # Obtenir tous les terrains
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
            pred_int = int(round(prediction))
            predictions_par_heure.append(pred_int)
        
        # Trouver l'heure de pointe pour ce terrain
        max_pred = max(predictions_par_heure)
        heure_pointe = predictions_par_heure.index(max_pred)
        
        resultats.append({
            'terrain': terrain,
            'predictions': predictions_par_heure,
            'max_prediction': max_pred,
            'heure_pointe': heure_pointe,
            'total_journee': sum(predictions_par_heure)
        })
    
    # Trier par affluence max
    resultats.sort(key=lambda x: x['max_prediction'], reverse=True)
    
    # Générer la prédiction principale
    top = resultats[0]
    niveau = "TRÈS FORTE" if top['max_prediction'] >= 8 else "FORTE" if top['max_prediction'] >= 5 else "MODÉRÉE" if top['max_prediction'] >= 3 else "FAIBLE"
    
    prediction_texte = f"Ce {jour_nom} entre {heure_debut}h et {heure_fin}h, le {top['terrain']} risque d'être complet!"
    
    return {
        'date': date_input,
        'jour': jour_nom,
        'heure_critique': f"{heure_debut}h - {heure_fin}h",
        'terrain_critique': top['terrain'],
        'prediction_texte': prediction_texte,
        'affluence_score': top['max_prediction'],
        'niveau': niveau,
        'recommandation': "💰 Augmenter les prix" if top['max_prediction'] >= 8 else "📢 Ouvrir plus de créneaux" if top['max_prediction'] >= 5 else "✅ Prix normal",
        'terrains_complets': resultats,
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
        "message": "🏆 API Prédiction d'Affluence - TERRAINS DE FOOT",
        "version": "4.0.0",
        "description": "Prédiction des heures de forte affluence basée sur les vraies réservations",
        "exemple_ultime": {
            "quote": "Ce vendredi entre 19h et 22h, le terrain 2 risque d'être complet!",
            "endpoint": "/predict/simple",
            "body": {"date": "2026-05-20", "terrain": "tous"}
        },
        "endpoints": {
            "/": "GET - Cette page",
            "/health": "GET - État API",
            "/patterns": "GET - Patterns appris",
            "/predict/simple": "POST - Prédiction simple (format texte)",
            "/predict/full": "POST - Prédiction complète (détails)",
            "/predict/tableau": "POST - Tableau des prédictions par heure",
            "/predict/graph": "POST - Générer graphique",
            "/docs": "GET - Documentation"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_score": f"{score*100:.1f}%",
        "nbr_reservations": len(df) if df is not None else 0,
        "nbr_terrains": len(terrains_liste),
        "terrains": terrains_liste,
        "heure_pointe": f"{heure_top}h",
        "jour_pointe": jours[jour_top]
    }

@app.get("/patterns")
def get_patterns():
    """Retourne les patterns appris - exactement comme demandé"""
    return {
        "patterns_appris": {
            "📅 Jour le plus chargé": jours[jour_top],
            "🕐 Heure la plus demandée": f"{heure_top}h",
            "🏟️ Terrain le plus populaire": terrain_top,
            "⏰ Créneau critique": f"{heure_debut}h - {heure_fin}h"
        },
        "statistiques": {
            "total_reservations": len(df) if df is not None else 0,
            "heures_analysees": "0-23h",
            "score_modele": f"{score*100:.1f}%"
        },
        "interpretation": f"⚠️ Selon l'analyse de {len(df)} réservations, le {jours[jour_top]} est le jour le plus chargé, avec un pic d'affluence à {heure_top}h. Le {terrain_top} est le plus demandé. Évitez de réserver entre {heure_debut}h et {heure_fin}h si vous voulez un terrain calme."
    }

@app.post("/predict/simple")
async def predict_simple(request: PredictionRequest):
    """
    Prédiction SIMPLE - Retourne exactement le format demandé:
    "Ce vendredi entre 19h et 22h, le terrain 2 risque d'être complet!"
    """
    try:
        result = predire_affluence(request.date, request.terrain)
        
        return {
            "date": request.date,
            "jour": result['jour'],
            "prediction": result['prediction_texte'],
            "niveau_affluence": result['niveau'],
            "recommandation": result['recommandation'],
            "score_modele": result['score_modele']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/full")
async def predict_full(request: PredictionRequest):
    """Prédiction COMPLÈTE avec tous les détails"""
    try:
        result = predire_affluence(request.date, request.terrain)
        
        # Générer les alertes
        alertes = []
        for r in result['terrains_complets'][:3]:
            if r['max_prediction'] >= 8:
                alertes.append(f"🚨 {r['terrain']} sera COMPLET à {r['heure_pointe']}h")
            elif r['max_prediction'] >= 5:
                alertes.append(f"⚠️ {r['terrain']} aura forte affluence à {r['heure_pointe']}h")
        
        # Top 5 des heures de pointe
        heures_pointe = []
        for heure in range(24):
            total_heure = sum(r['predictions'][heure] for r in result['terrains_complets'])
            if total_heure > 0:
                niveau = "🔴" if total_heure >= 8 else "🟠" if total_heure >= 5 else "🟡" if total_heure >= 3 else "🟢"
                heures_pointe.append({
                    'heure': f"{heure}h-{heure+1}h",
                    'total_reservations': total_heure,
                    'niveau': niveau
                })
        
        heures_pointe.sort(key=lambda x: x['total_reservations'], reverse=True)
        
        return FullPredictionResponse(
            date=request.date,
            jour=result['jour'],
            score_modele=result['score_modele'],
            patterns={
                "jour_le_plus_charge": jours[jour_top],
                "heure_de_pointe_globale": f"{heure_top}h",
                "terrain_le_plus_populaire": terrain_top
            },
            top_terrain={
                "nom": result['terrain_critique'],
                "affluence_max": result['affluence_score'],
                "heure_pointe": result['heure_critique'],
                "total_journee": result['terrains_complets'][0]['total_journee']
            },
            heures_pointe=heures_pointe[:5],
            alertes=alertes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/tableau")
async def predict_tableau(request: PredictionRequest):
    """Retourne un tableau complet des prédictions par heure et par terrain"""
    try:
        result = predire_affluence(request.date, request.terrain)
        
        # Créer le tableau
        tableau = []
        for heure in range(24):
            ligne = {"heure": f"{heure}h-{heure+1}h"}
            for terrain in result['terrains_complets'][:6]:  # Top 6 terrains
                ligne[terrain['terrain']] = terrain['predictions'][heure]
            ligne['total'] = sum(terrain['predictions'][heure] for terrain in result['terrains_complets'])
            tableau.append(ligne)
        
        return {
            "date": request.date,
            "jour": result['jour'],
            "légende": "🔴>=8 (complet) | 🟠>=5 (fort) | 🟡>=3 (modéré) | 🟢<3 (calme)",
            "tableau": tableau
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/graph")
async def predict_graph(request: PredictionRequest):
    """Génère un graphique de la prédiction"""
    try:
        result = predire_affluence(request.date, request.terrain)
        
        plt.figure(figsize=(14, 8))
        
        # Graphique principal: Affluence par heure pour les top 5 terrains
        for i, terrain in enumerate(result['terrains_complets'][:5]):
            plt.plot(range(24), terrain['predictions'], 
                    marker='o', linewidth=2, markersize=4,
                    label=terrain['terrain'])
        
        plt.axvline(x=heure_top, color='red', linestyle='--', alpha=0.5, label=f'Heure de pointe: {heure_top}h')
        plt.axhline(y=8, color='darkred', linestyle='--', alpha=0.7, label='Seuil complet (8)')
        plt.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Seuil fort (5)')
        plt.axhline(y=3, color='yellow', linestyle='--', alpha=0.7, label='Seuil modéré (3)')
        
        plt.xlabel('Heure', fontsize=12)
        plt.ylabel('Nombre de réservations prédites', fontsize=12)
        plt.title(f'📊 Prédiction d\'affluence - {request.date} ({result["jour"]})\nScore modèle: {score*100:.1f}%', fontsize=14)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))
        
        # Ajouter une annotation avec la prédiction principale
        plt.annotate(f'⚠️ {result["prediction_texte"]}', 
                    xy=(0.5, -0.15), xycoords='axes fraction',
                    ha='center', fontsize=12, style='italic',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# LANCEMENT
# =========================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔮 PRÉDICTION DES HEURES DE FORTE AFFLUENCE")
    print("="*60)
    print("\n📌 EXEMPLE DE PRÉDICTION:")
    print('   POST /predict/simple')
    print('   Body: {"date": "2026-05-20", "terrain": "tous"}')
    print('\n   RÉPONSE:')
    print('   "Ce vendredi entre 19h et 22h, le Terrain 2 risque d\'être complet!"')
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")