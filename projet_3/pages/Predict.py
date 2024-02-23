import streamlit as st
import pandas as pd
import numpy as np
import missingno as msno
import streamlit_extras
import Utils.Utils as u

u.init_page("Predict")
if("model" in st.session_state):
    model = st.session_state["model"]
    file = st.file_uploader(label="Mettez votre CSV",type="csv",accept_multiple_files=False)
    txt_sep = st.text_input("Séparateur",",",max_chars=3)
    bt_predict = st.button("Traiter")
    if file and bt_predict:
        sep = ","
        if (txt_sep != ""):
            sep = txt_sep 
        data_target = pd.read_csv(file,sep=sep)
        if("target" in data_target.columns):
            data_target.drop("target",axis=1,inplace=True)
        data_target["prediction"] = model.predict(data_target)
        st.success("Votre fichier " + str(file.name) + " a été correctement importé et évalué. Voici un aperçu des données :",icon="✅" )
        st.dataframe(data_target.head())
        st.session_state["data_predict"] = data_target
    if "data_predict" in st.session_state:
        data_target = st.session_state["data_predict"]
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        csv = convert_df(data_target)

        st.download_button(
        "Télécharger les données",
        csv,
        "pred_"+st.session_state["nom_dataset"]+".csv",
        "text/csv",
        key='download-csv'
        )
else:
    st.warning("Vous devez charger ou entrainer un modèle")