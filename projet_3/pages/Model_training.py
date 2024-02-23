import streamlit as st
import pandas as pd
import Utils.Utils as u
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings

from pylab import *
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn import preprocessing 
import seaborn as sns
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, mean_squared_error,r2_score
from sklearn import model_selection
from sklearn.linear_model import *
from sklearn.svm import SVC 
from sklearn.svm import SVR

def encode_column(df,col,verbose):
    if df[col].dtype == 'object':
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        if verbose:
            st.text("Colonne " + col + " réencodée automatiquement")
    else:
        return col

u.init_page("Entrainement")

#On récupère le dataframe en session
if("data" in st.session_state):
    data = st.session_state["data"]

    #Est-ce qu'on a une colonne target ? Sinon on propose de la créer
    if not "target" in data.columns:
        st.warning("Pas de colonne ""target"" dans les données vous devriez sélectionner une colonne pour la transformer en target, sinon il sera impossible de continuer")
        col_target = st.selectbox("Sélectionnez la target",data.columns)
        bt_target = st.button("Transformer en colonne Target")
        if(bt_target):
            data["target"] = data[col_target]
            data = data.drop(col_target,axis=1)

    #Si on a bien une colonne target on propose de faire un split perso
    if "target" in data.columns:
        y = data["target"]
        X = data.drop("target",axis=1)
        if("train_test" in st.session_state):
            X_train, X_test, y_train, y_test = st.session_state["train_test"][0],st.session_state["train_test"][1],st.session_state["train_test"][2],st.session_state["train_test"][3]
        with st.expander("Données",True):
            st.text("La colonne target contient "+str(data["target"].value_counts().count())+" valeurs différentes, voici le top 5:")
            st.dataframe(data["target"].value_counts().head())
            st.text("Le jeu de données contient "+str(data["target"].count()) + " lignes")
            split = st.slider("Split Train / Test",0,100,33,1)
            generer_split = st.button("Découper")

            #Génération du split si le bouton a été utilisé
            if(generer_split):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split/100.0, random_state=42)
                st.text(str(len(X_train)) + " lignes d'entrainement")
                st.text(str(len(X_test)) + " lignes de test")
                st.success("Split initialisé")
                st.session_state["train_test"] = (X_train, X_test, y_train, y_test) 

    with st.expander("Modèle"):
        #as-t-on créé un split personnalisé ? sinon on va en initialiser un
        try:
            __test = X_train
            init_train = False
        except NameError:
            init_train = True

        #Un onglet par type de ML
        tab_reg,tab_clas,tab_clus,tab_dim = st.tabs(["Regression","Classification","Clustering","Réduction des dimensions"])
        with tab_reg:
            algo = st.selectbox("Choisissez un modèle",("","Régression","Régression Lasso","Régression Ridge","ElasticNet","SVR"))
            warnings.filterwarnings("ignore")
            if algo != "":
                if init_train:
                        st.text("Aucun jeu de test créé, initialisé avec des paramètres par défaut")
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
                        st.text(str(len(X_train)) + " lignes d'entrainement")
                        st.text(str(len(X_test)) + " lignes de test")
                else:
                    st.text("Le jeu de données de test personnalisé a été utilisé")

                bt_trainr = st.button("Entrainer le modèle",key="bt_trainr")
                if bt_trainr:
                    st.subheader("Génération du modèle " + algo)
                    if algo == "Régression Lasso":
                        model = Lasso()
                    if algo == "Régression":
                        model = LinearRegression()
                    if algo == "Régression Ridge":
                        model = Ridge()
                    if algo == "ElasticNet":
                        model = ElasticNet()
                    if algo =="SVR":
                        model = SVR()
                    
                    for col in X_train.columns:
                        encode_column(X_train,col,True)
                    for col in X_test.columns:
                        encode_column(X_test,col,False)
                        
                    model.fit(X_train, y_train)
                    y_hat=model.predict(X_test)
                    mse = round(mean_squared_error(y_test,y_hat),4)
                    r2 = r2_score(y_test,y_hat)

                    st.success("Modèle entrainé , R2 score = " + str(r2) + " , MSE = " + str(mse))

                    st.session_state["Data"] = data
                    st.session_state["model"] = model
                    st.success("Modèle sauvegardé !")
        with tab_clas:
            algo = st.selectbox("Choisissez un modèle",("","Régression Logistique","K Nearest Neighbours","SVC"))
            if algo != "":
                
                
                if init_train:
                    st.text("Aucun jeu de test créé, initialisé avec des paramètres par défaut")
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
                    st.text(str(len(X_train)) + " lignes d'entrainement")
                    st.text(str(len(X_test)) + " lignes de test")
                else:
                    st.text("Le jeu de données de test personnalisé a été utilisé")
                if(algo == "K Nearest Neighbours"):
                    txt_voisins = st.text_input("Nombre de voisins")
                bt_trainc = st.button("Entrainer le modèle",key="bt_trainc")
                if bt_trainc:
                    if algo == "Régression Logistique":
                        st.subheader("Génération du modèle Regression Logistique")
                        model = LogisticRegression()
                    if algo == "K Nearest Neighbours":
                        st.subheader("Génération du modèle KNN")
                        if str(txt_voisins).isdigit():
                            nb_voisins = int(txt_voisins)
                        else:
                            nb_voisins = 10
                        if nb_voisins>len(X_train):
                            nb_voisins = int(np.floor(len(X_train)/2))
                            st.text("nombre de nb_voisins ajusté à " + str(nb_voisins))
                        if nb_voisins>len(X_test):
                            nb_voisins = int(np.floor(len(X_test)/2))
                            st.text("nombre de nb_voisins ajusté à " + str(nb_voisins))
                        model = neighbors.KNeighborsClassifier(n_neighbors = nb_voisins)
                    if algo == "SVC":
                        st.subheader("Génération du modèle SVC")
                        model = SVC(kernel='linear')
                    with st.spinner("Veuillez patienter pendant l'entrainement du modèle"):
                        

                        #--------------------------------------
                        my_kfold =    model_selection.KFold(n_splits=10, shuffle=True, random_state=0)
                        scores = model_selection.cross_val_score(estimator=model,
                                                X=X,
                                                y=y,
                                                cv=my_kfold,
                                                n_jobs=-1) 
                        st.text("Moyenne du score du modèle: " + str(mean(scores)))
                        from sklearn.model_selection import cross_val_score, KFold
                    st.text("Matrice de confusion : ")
                    model.fit(X_train, y_train)
                    predictions_log=model.predict(X_test)
                    cm = confusion_matrix(y_test, predictions_log)
                    fig, ax = plt.subplots()
                    sns.set_theme(font_scale=1.2)  # Adjust font size
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title('Confusion Matrix')
                    st.write(fig)

                    # Generate classification report
                    report_dict = classification_report(y_test, predictions_log, output_dict=True)

                    # Convert classification report to DataFrame
                    report_df = pd.DataFrame(report_dict)

                    # Display classification report in Streamlit
                    st.write("Classification Report:")
                    st.dataframe(report_df)
                    st.session_state["Data"] = data
                    st.session_state["model"] = model
                    st.success("Modèle sauvegardé !")
        with tab_clus:
            st.warning("WIP : Cette fonctionnalité sera disponible prochainement",icon="🚧")
        with tab_dim:
            st.warning("WIP : Cette fonctionnalité sera disponible prochainement",icon="🚧")

else:
    st.text("Bienvenue, allez dans file upload pour charger un CSV")