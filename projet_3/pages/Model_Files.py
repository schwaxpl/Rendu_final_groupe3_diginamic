import streamlit as st
import pandas as pd
import Utils.Utils as u
import joblib
import io
import time as t
u.init_page("Import/export modèle")

def convert_mod(_model_to_dump):
    fname=""
    with open("model"+u.get_model_name()+".pkl","wb") as file2:
        joblib.dump(_model_to_dump,file2)
        fname = file2.name
    return fname
file = st.file_uploader(label="Mettez votre modèle",type="pkl",accept_multiple_files=False)

if file:
    model = joblib.load(file)
    st.text("Modèle chargé avec succès !")
    st.session_state["model"] = model

if "model" in st.session_state:
    model = st.session_state["model"]

    filename = convert_mod(model)

    with open(filename, "rb") as file3:
        model_bytes = file3.read()
        st.download_button(
        "Télécharger le modèle " + u.get_model_name(),
        model_bytes,
        file3.name + str(t.time()),
        mime="application/octet-stream",
        key='download-pkl'+u.get_model_name()+str(t.time())
        )
