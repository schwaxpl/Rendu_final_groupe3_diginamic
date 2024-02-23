import streamlit as st
import pandas as pd
import numpy as np
import missingno as msno
import streamlit_extras
import Utils.Utils as u

msg = ''
if 'clean_message' in st.session_state :
    msg = st.session_state["clean_message"]
def save_data(data,message):
    st.session_state["data"] = data
    st.session_state["clean_message"] = message
    st.rerun()

u.init_page("Nettoyage")

st.title('Nettoyage des données')
if msg != '':
    st.success(msg)
    del st.session_state["clean_message"]

if("data" in st.session_state):
    data = st.session_state["data"]

    # Affichage des colonnes
    with st.expander('Information des colonnes'):
        st.markdown('## Information des colonnes')
        display_option = st.radio('Choisissez une option', ['Afficher toutes les colonnes', 'Sélectionner des colonnes'],key='display_option')

        if display_option == 'Afficher toutes les colonnes':
            st.subheader('Dataframe complet')
            st.dataframe(data.head())
            
            
        else:
            selected_columns = st.multiselect('Sélectionnez les colonnes à afficher', data.columns,key='selected_columns')

            if selected_columns:
                st.subheader('Dataframe avec les colonnes sélectionnées')
                st.write(data[selected_columns])
            else:
                st.warning('Veuillez sélectionner au moins une colonne.')

        list_columns = list(data.columns)
        onglet_info = st.tabs(list_columns)

        for idx, col_name in enumerate(list_columns):
            with onglet_info[idx]:
                st.text(f'Nombre de valeurs non nulles : {data[col_name].count()}')
                st.text(f'Type de données : {data[col_name].dtype}')
                st.text(f'Nombre de valeurs uniques : {data[col_name].nunique()}')

    # Choix des colonnes
    with st.expander('Choix et Renommage des colonnes'):
        st.markdown('## Choix des colonnes')
        columns_to_drop = st.multiselect('Sélectionnez les colonnes inutiles à supprimer', data.columns,key='columns_to_drop')

        if st.button('Supprimer les colonnes sélectionnées',key='button_drop_columns'):
            if columns_to_drop:
                data = data.drop(columns=columns_to_drop, axis=1)
                save_data(data,'Colonne(s) supprimé(s) avec succès.')
            else:
                st.warning('Veuillez sélectionner au moins une colonne à supprimer.')

            # Bouton pour revenir en arrière
        undo_columns_button_id = 'undo_columns_button'
        if st.button('Revenir en arrière', key=undo_columns_button_id):
            data = pd.DataFrame(data)
            save_data(data,'Opération annulée. Dataframe réinitialisé.')

        st.subheader('Dataframe mis à jour')
        st.dataframe(data.head())

        # Renommer une colonne 
        st.markdown('### Renommer des colonnes')
        selected_column = st.selectbox('Choisissez une colonne à renommer:', st.session_state['data'].columns, key='rename_column_select')
        new_column_name = st.text_input('Nouveau nom de la colonne:', key='new_column_name_input')
        if st.button('Changer le nom de la colonne', key='rename_column_button'):
            if new_column_name:
                data.rename(columns={selected_column: new_column_name}, inplace=True)
                st.session_state["data"] = data
                save_data(data,f'Nom de la colonne "{selected_column}" changé avec succès en "{new_column_name}".')


            st.subheader('Dataframe mis à jour')
            st.dataframe(data.head())

    # Imputation / Remplacement des NaN
    with st.expander('Imputation et remplacement des valeurs manquantes'):
        st.markdown('## Imputation des valeurs manquantes')
        st.markdown('### Information des valeurs manquantes :')
        missing_values = data.isnull().sum()
        st.table(missing_values.reset_index().rename(columns={0: 'Nombre de valeurs manquantes'}).style.highlight_null()) # Affichage des Nan
        st.set_option('deprecation.showPyplotGlobalUse', False) 
        msno.matrix(data)
        st.pyplot()

        st.markdown('### Remplacer / supprimer des valeurs manquantes :')
        selected_column = st.selectbox('Sélectionnez une colonne:', data.select_dtypes('number').columns,key='key_drop_replace_nan')
        replace_button_id_median = f'replace_button_median_{selected_column}'
        replace_button_id_specific = f'replace_button_specific_{selected_column}'
        return_button_id = f'return_button_{selected_column}'
        replace_button_id_dropna = f'replace_button_dropna_{selected_column}'

        replacement_value = st.text_input('Nouvelle valeur de remplacement:', '')
        if st.button('Remplacer les valeurs manquantes', key='button_replace_nan_id'):
            data[selected_column].fillna(replacement_value, inplace=True)
            save_data(data,f'Valeurs manquantes dans la colonne "{selected_column}" remplacées avec succès.')

        if st.button('Remplacer les valeurs manquantes par la médiane', key=replace_button_id_median):
            median_value = data[selected_column].median()
            data[selected_column].fillna(median_value, inplace=True)
            save_data(data,f'Valeurs manquantes dans la colonne "{selected_column}" remplacées avec succès.')

        if st.button('Supprimer les valeurs manquantes restantes',key=replace_button_id_dropna):
            data.dropna(inplace=True)
            save_data(data,'Valeurs manquantes restantes supprimées avec succès.')

        if st.button('Revenir en arrière', key=return_button_id):
            data = pd.DataFrame(data)
            save_data(data,'Opération annulée. Dataframe réinitialisé.')

        st.subheader('Dataframe mis à jour')
        st.dataframe(data.head())


        missing_values = data.isnull().sum()
        st.table(missing_values.reset_index().rename(columns={0: 'Nombre de valeurs manquantes'}).style.highlight_null())

    # Transformation des données
    with st.expander('Transformation des données'):
        st.markdown('## Transformation des données')

        # Garder des motifs dans les valeurs
        st.markdown('### Garder des motifs dans les valeurs')
        st.subheader('Dataframe original')
        st.dataframe(data.head())

        selected_column = st.selectbox('Sélectionnez une colonne:', data.columns,key='key_gader_motifs')

        selected_pattern = st.text_input('Entrez le motif à rechercher dans les valeurs:', key='key_input_pattern')
        keep_option = st.radio('Choisissez ce qui sera gardé:', ['Avant le motif', 'Après le motif'], key='key_radio_keep_option')
        apply_to_same_column = st.checkbox('Appliquer la transformation à la même colonne', key='key_checkbox_apply')
        new_column_name = st.text_input('Nom de la nouvelle colonne (si différent de la colonne d\'origine):', key='key_text_input_new_column')

        def custom_transform(value, selected_pattern, keep_option):
            if isinstance(value, (str, int, float)):
                words = str(value).split()
                if selected_pattern.lower() in [word.lower() for word in words]:
                    pattern_index = [word.lower() for word in words].index(selected_pattern.lower())
                    if keep_option == 'Avant le motif':
                        return ' '.join(words[:pattern_index])
                    elif keep_option == 'Après le motif':
                        return ' '.join(words[pattern_index + 1:])
            return value
        
        if st.button('Appliquer la transformation', key='key_button_apply_transform'):
            if apply_to_same_column or new_column_name:
                transformed_column = selected_column if apply_to_same_column else new_column_name
                data[transformed_column] = data[selected_column].apply(lambda x: custom_transform(x, selected_pattern, keep_option))
                save_data(data,'La transformation des motifs gardés est un succès ')
                st.subheader(f'Dataframe après transformation: {transformed_column}')
                st.dataframe(data.head())
            else:
                st.error("Veuillez fournir un nom pour la nouvelle colonne.")


        # Remplacer des valeurs
        st.markdown('### Remplacement de valeurs')
        selected_column = st.selectbox('Sélectionnez une colonne:', data.columns,key='key_replace_val')

        unique_values = data[selected_column].unique()
        st.write(f"Valeurs uniques dans la colonne '{selected_column}':")
        st.write(unique_values)

        value_to_replace = st.text_input('Valeur à remplacer:', '')
        new_value = st.text_input('Nouvelle valeur:', '')

        if st.button('Remplacer les valeurs',key='button_validate_transform_data'):
            if value_to_replace != '' and new_value != '':
                data[selected_column] = data[selected_column].replace(value_to_replace, new_value)
                save_data(data,f'Valeurs dans la colonne "{selected_column}" remplacées avec succès.')
            else:
                st.warning('Veuillez entrer une valeur à remplacer et une nouvelle valeur.')

            st.subheader('Dataframe mis à jour')
            st.dataframe(data.head())

else:
    st.text("Bienvenue, allez dans file upload pour charger un CSV")