import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

import Utils.Utils as u

u.init_page("Statistiques")

nb_catcolumn=0
nb_numcolumn=0

if("data" in st.session_state):
    data = st.session_state["data"]

    with st.expander("Column Type"):    
        st.subheader("Column Data Types:") # Iterate over each column and display if it's categorical or numerical
        for column_name in data.columns:
            if data[column_name].dtype == 'object':
                nb_catcolumn+=1
            else:
                nb_numcolumn+=1
        st.write(f"The dataset contains {nb_catcolumn} categorical columns and " f"{nb_numcolumn} numerical columns")

        st.subheader("Data types of each column:") # Check data types of each column
        st.write(data.dtypes)
    
    if nb_catcolumn !=0:    
        with st.expander("Matrice de correlation"):
            data_numeric = data.apply(pd.to_numeric, errors='coerce')
            st.dataframe(data_numeric.corr())
            fig, ax = plt.subplots()
            sns.heatmap(data_numeric.corr(), ax=ax,)
            st.write(fig)
    else:
        with st.expander("Matrice de correlation"):
            st.dataframe(data.corr())
            fig, ax = plt.subplots()
            sns.heatmap(data.corr(), ax=ax,)
            st.write(fig)

    with st.expander("Visualization de données"):
        pplot= st.button("Exploration des données Pairplot")
        histoplot= st.button("Représentation des données par histogramme")
        
        if pplot or histoplot:
            st.subheader("Visualization")
            if pplot:
                if("data" in st.session_state):
                    data = st.session_state["data"]
                    st.title('Pairplot de chacune des variables')
                    for column_x in data.columns:
                        if column_x != 'target':
                            for column_y in data.columns:
                                if column_y != 'target' and column_x != column_y:
                                    st.write(f'Pairplot for variables {column_x} and {column_y}')
                                    pairplot = sns.pairplot(data, x_vars=column_x, y_vars=column_y, hue="target")
                                    st.pyplot(pairplot.fig)


                    st.pyplot(sns.pairplot(data,hue="target"))
            if histoplot:
                if("data" in st.session_state):
                    data = st.session_state["data"]
                    st.title('Histogrammes de chacune des variables')
                    for column in data.columns:
                        if column != 'target':  
                            st.write(f'Histogramme de la variable {column}')
                            fig, ax = plt.subplots(figsize=(6, 6))
                            data[column].hist(bins=25, ax=ax)
                            ax.set_xlabel(column)
                            ax.set_ylabel('Fréquence')
                            st.pyplot(fig)
    
    with st.expander("Description des données"):
        st.dataframe(data.describe())

    with st.expander("Informations"):
        if("data" in st.session_state):
            data = st.session_state["data"]
            st.write(data.info())

    with st.expander("Usage mémoire"):
        st.dataframe(data.memory_usage())

    
else:
    st.text("Bienvenue, allez dans file upload pour charger un CSV")

