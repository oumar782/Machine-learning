import streamlit as st
import pandas as pd
import app as prediction_app

st.set_page_config(
    page_title="Prédiction d'affluence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_app():
    prediction_app.load_and_train()
    return prediction_app

prediction_module = initialize_app()

st.title("Prédiction d'affluence de terrain")
st.write(
    "Utilisez ce modèle pour estimer les heures de pointe et l'affluence d'un terrain en fonction de la date et du type de terrain."
)

with st.sidebar:
    st.header("Paramètres")
    date_input = st.date_input("Date de réservation", value=pd.Timestamp.today())
    terrain_options = ["tous"] + prediction_module.terrains_liste
    terrain_input = st.selectbox("Terrain", terrain_options)
    run_button = st.button("Lancer la prédiction")

if run_button:
    with st.spinner("Calcul en cours…"):
        result = prediction_module.predire_affluence_avec_date(
            date_input.strftime("%Y-%m-%d"), terrain_input
        )

    st.subheader("Résumé de la prédiction")
    st.metric("Type de jour", result["type_jour"])
    st.metric("Coefficient d'affluence", result["coefficient"])
    st.metric("Terrain critique", result["terrain_critique"])
    st.metric("Heure critique", result["heure_critique"])
    st.markdown(f"**Conseil :** {result['recommandation']}")

    st.markdown("---")
    st.subheader("Affluence horaire estimée")

    hours = list(range(24))
    data = {
        terrain["terrain"]: terrain["predictions"]
        for terrain in result["terrains"]
    }
    df_chart = pd.DataFrame(data, index=hours)
    st.line_chart(df_chart)

    st.subheader("Détails par terrain")
    st.dataframe(
        pd.DataFrame(result["terrains"]).rename(
            columns={
                "terrain": "Terrain",
                "max_prediction": "Affluence max",
                "heure_pointe": "Heure de pointe",
                "total_journee": "Total journalier"
            }
        )
    )
else:
    st.info("Choisissez une date et un terrain, puis cliquez sur \"Lancer la prédiction\".")
