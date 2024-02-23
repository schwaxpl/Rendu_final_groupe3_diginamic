import streamlit as st
import pandas as pd
import numpy as np
import missingno as msno
import streamlit_extras
import Utils.Utils as u
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler,MinMaxScaler

msg = ''
if 'clean_message' in st.session_state :
    msg = st.session_state["clean_message"]
def save_data(data,message):
    st.session_state["data"] = data
    st.session_state["clean_message"] = message
    st.rerun()

u.init_page("Encodage_Standardisation")

st.title('Encodage et Standardisation des données')
if msg != '':
    st.success(msg)

if("data" in st.session_state):
    data = st.session_state["data"]

    # Encodage des données
    with st.expander('Encodage'):
        st.markdown('## Encodage des données')

        def detect_categorical_type(series):
                unique_values = series.unique()
                num_unique = len(unique_values)
                num_missing = series.isnull().sum()

                if num_unique <= 10:  
                    return 'Nominal'
                elif num_unique > 10 and num_missing == 0: 
                    return 'Ordinal'
                else:
                    return 'Non Categorique'
                
        nb_catcolumn = 0
        nb_numcolumn = 0

        st.subheader("Column Data Types:")
        for column_name in data.columns:
            if data[column_name].dtype == 'object':
                nb_catcolumn += 1
            else:
                nb_numcolumn += 1

        st.write(f"The dataset contains {nb_catcolumn} categorical columns and {nb_numcolumn} numerical columns")
        st.subheader("Data types of each column:")
        st.write(data.dtypes)

        categorical_columns = [col for col in data.columns if detect_categorical_type(data[col]) == 'Nominal' and data[col].dtype == 'object']
        selected_categorical_columns = st.multiselect('Sélectionnez les colonnes catégoriques à encoder', categorical_columns, key='selected_categorical_columns')

        encode_button_id = 'encode_button'
        if st.button('Confirmer l\'encodage', key=encode_button_id):
            for col in selected_categorical_columns:
                encoding_method = 'Ordinal Encoder' if detect_categorical_type(data[col]) == 'Ordinal' else 'One Hot Encoder'
                if encoding_method == 'Ordinal Encoder':
                    encoder = OrdinalEncoder()
                else:
                    encoder = OneHotEncoder(drop='first')
                
                col_enc = encoder.fit_transform(data[[col]])
                encode_df = pd.DataFrame(col_enc.toarray())
                for idx, i in enumerate(encode_df.columns):
                    encode_df.rename(columns={i: encoder.get_feature_names_out()[idx]}, inplace=True)
                data = pd.concat([data, encode_df], axis=1)
                data = data.drop(columns=col)

            save_data(data,'Colonnes encodées avec succès.')
            
        st.subheader('Dataframe mis à jour après l\'encodage')
        st.dataframe(data.head())

    # Standardisation des données
    with st.expander('Standardisation'):
        st.markdown('## Standardisation des données')

        selected_columns = st.multiselect('Sélectionnez les colonnes à standardiser:', data.select_dtypes(include=['float64', 'int64']).columns)

        # Bouton de standardisation au Z score
        if selected_columns:
            if st.button('Standardiser au Z score', key='button_standard_scaler'):
                scaler = StandardScaler()
                data[selected_columns] = scaler.fit_transform(data[selected_columns])
                save_data(data,f'Colonnes {selected_columns} standardisées avec StandardScaler.')

        # Bouton de standardisation au min/max
        if selected_columns:
            if st.button('Standardiser au mininum / maximum', key='button_min_max_scaler'):
                scaler = MinMaxScaler()
                data[selected_columns] = scaler.fit_transform(data[selected_columns])
                save_data(data,f'Colonnes {selected_columns} appliquées avec succès Min-Max Scaling.')

        st.subheader('Dataframe mis à jour')
        st.dataframe(data.head())

else:
    st.text("Bienvenue, allez dans file upload pour charger un CSV")