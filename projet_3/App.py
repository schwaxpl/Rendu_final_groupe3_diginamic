import streamlit as st
import pandas as pd
import numpy as np
from Utils import Utils as u

u.init_page("Main")

if("data" in st.session_state):
    data = st.session_state["data"]
    st.text("Voici les données chargées en mémoire :")


    # Set the number of rows to display per page
    rows_per_page = 10

    # Calculate the total number of pages
    total_pages = -(-len(data) // rows_per_page)  # Round up division

    # Add a selectbox for selecting the page
    page_index = st.selectbox("Page", range(total_pages))

    # Calculate the start and end row index for the current page
    start_row = page_index * rows_per_page
    end_row = min((page_index + 1) * rows_per_page, len(data))

    # Slice the DataFrame to display the current page
    paginated_data = data.iloc[start_row:end_row]

    # Display the paginated DataFrame
    st.write(paginated_data)
else:
    st.text("Bienvenue, allez dans file upload pour charger un CSV")