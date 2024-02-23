import streamlit as st
import pandas as pd
import Utils.Utils as u
from pylab import *
from sklearn.svm import SVC 

from sklearn.decomposition import PCA
from sklearn import preprocessing 
import seaborn as sns
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import model_selection
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, KFold



u.init_page("Extras")

with st.expander("Traitement pré configuré ""Vin"" "):
    bt_explo = st.button("Exploration des données pour classification")
    bt_vin = st.button("Générer un modèle KNearest Neighbors",key="vin")
    bt_vin2 = st.button("Générer un modèle de regression linéaire",key="vin2")
    bt_vin3 = st.button("Générer un modèle SVC",key="vin3")

    if("data" in st.session_state):
        data = st.session_state["data"]
        X=data[data.columns[1:-1]]
        Y=data["target"]
        X = preprocessing.scale(X)
        features_all_train, features_all_test, activity_all_train, activity_all_test = model_selection.train_test_split(X,Y,train_size=0.7,random_state=42)
    
    else:
        st.text("Vous devez d'abord importer un fichier")

    if bt_explo or bt_vin or bt_vin2 or bt_vin3:
        st.subheader("Résultats")
        if bt_explo:
            if("data" in st.session_state):
     
                if("data" in st.session_state):
                    data = st.session_state["data"]
                    X=data[data.columns[1:-1]]
                    Y=data["target"]
                    st.text("Votre variable taget a "+ str(len(np.unique(data.iloc[:,-1])))+" catégories :")
                    A=pd.DataFrame(data.iloc[:,-1].value_counts())
                    A.rename(columns={A.columns[0]:"quantité par catégorie"})
                    A.index.name="Catégories dans la variable cible"
                    st.dataframe(A)
                    a=np.sum(data.isna()) # pas de na
                    if np.all(a==0):
                        st.text("Il n'y a pas de données manquantes ! ")
                    else:
                        st.text("Il y a ",np.sum(a), "données manquantes",a)

                    z=np.where(data.duplicated())[0] # Pas de lignes dupliquées 

                    if z==0:
                        st.text("Il n'y a pas des lignes dupliquées ! ")
                    else:
                        st.text("Il y a " +str(len(z)) +" lignes dupliquées")

                    type=pd.DataFrame(data.dtypes.value_counts())
                    type.index.name="data type"
                    st.text("Vos données ont les types suivants :")
                    st.dataframe(type)
            
                    st.text("Matrice de correlation : ")
                    correlation_matrix = pd.DataFrame(X).corr()
                    correlation_matrix=correlation_matrix[correlation_matrix>0.5]
                    mask = np.triu(correlation_matrix.select_dtypes("number").corr())
                    correlation_matrix.index=data.columns[1:-1]
                    correlation_matrix.columns=data.columns[1:-1]
                    # Tracé de la heatmap
                    fig, ax = plt.subplots()
                    sns.heatmap(correlation_matrix, mask=mask,annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5,ax=ax)
                    st.write(fig)

            #st.pyplot(plt.barh(width=data.iloc[:,-1].value_counts(),y=data.iloc[:,-1].value_counts().index))
                    with st.spinner("Veuillez patienter pendant la génération du graphique"):
                        st.pyplot(sns.pairplot(data,hue="target"))
            else:
                st.text("Veuillez importer un dataset")

        if bt_vin:
            if("data" in st.session_state):
                data = st.session_state["data"]
                X=data[data.columns[1:-1]]
                Y=data["target"]

                
                X = preprocessing.scale(X)
                st.text("Pourcentage jeu d'entrainement : " + str(len(activity_all_train)/ float(len(X))))
                st.text("Pourcentage jeu de test : " + str(len(activity_all_test)/ float(len(X))))

                st.title("Génération du modèle KNeighborsClassifier")
                #---------------
                with st.spinner("Veuillez patienter pendant le GridSearchCV"):
                    features_all_train, features_all_test, activity_all_train, activity_all_test = model_selection.train_test_split(X,Y,train_size=0.7,random_state=42)
                    from sklearn.model_selection import GridSearchCV
                    st.text("On va tester un nombre de voisins compris entre 2 et 20")
                    # la grille de paramètres a régler sont definis dans un dictionnaire (dict)
                    tuned_parameters = {'n_neighbors': range(2,20)}

                    my_kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=0)
                    nnGrid = GridSearchCV(neighbors.KNeighborsClassifier(),
                                        tuned_parameters,
                                        cv=5)
                    nnGrid.fit(features_all_train, activity_all_train)

                    # le meilleur modèle 
                    st.text('On choisit de conserver ' + str( nnGrid.best_params_['n_neighbors'])+ ' voisins.')
                    st.text("On a obtenu le meilleur score de " + str( nnGrid.best_score_) + " avec ce paramètres")

                #----------
                

                    st.text("Le taux de bon classement pour KN sur l'échantillon test est " + str(nnGrid.score(features_all_test,activity_all_test)))
                    st.text("Le taux de bon classement pour KN sur l'échantillon d'entrainement est " + str(nnGrid.score(features_all_train,activity_all_train)))
                
                with st.spinner("Veuillez Les résultats sur les 10 splits utilisés :"):
                
                    nn_val_croisee = neighbors.KNeighborsClassifier(n_neighbors = nnGrid.best_params_['n_neighbors'])

                    scores = model_selection.cross_val_score(estimator=nn_val_croisee,
                                            X=X,
                                            y=Y,
                                            cv=my_kfold,
                                            n_jobs=-1) # permet de répartir les calculs sur plusieurs cœurs
                    st.text("Scores sur les 10 splits utilisés : " + str(scores))
                    st.text("Moyenne du score : " + str(mean(scores)))

                    st.text("Matrice de confusion : ")
                    predictions = nnGrid.predict(features_all_test)
                    cm = confusion_matrix(activity_all_test, predictions)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    fig, ax = plt.subplots()
                    sns.set_theme(font_scale=1.2)  # Adjust font size
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title('Confusion Matrix')
                    st.write(fig)
                    st.session_state["Data"] = data
                    st.session_state["model"] = nnGrid.best_estimator_
                    st.text("Modèle sauvegardé !")
            else:
                st.text("Vous devez d'abord importer un fichier")

        if bt_vin3:
        
            if("data" in st.session_state):

                data = st.session_state["data"]
                X=data[data.columns[1:-1]]
                Y=data["target"]
                X = preprocessing.scale(X)
                features_all_train, features_all_test, activity_all_train, activity_all_test = model_selection.train_test_split(X,Y,train_size=0.7,random_state=42)

                from sklearn.model_selection import GridSearchCV
                from time import time

                start = time()

                my_kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state = 0)
                parameters = { 'degree':[1,2,3,4,5], 'C':[0.001,0.01,0.05,0.1,1,2,10,100,1000,10000],'gamma':['scale', 'auto']}
                svc = SVC(kernel='poly')
                svcgrid = GridSearchCV(svc, parameters,cv=5,n_jobs=-1)
                svcgrid.fit(features_all_train,activity_all_train)
                svc_pred=svcgrid.best_estimator_.predict(features_all_test)

                st.write("le score du prédiction sur l'échantillon  test avec les paramètres du GridsearchCV est : ",svcgrid.best_estimator_.score(features_all_test,activity_all_test))

                scores = model_selection.cross_val_score(estimator=svcgrid.best_estimator_,
                        X=X,
                        y=Y,
                        cv=my_kfold,
                        n_jobs=-1) # permet de répartir les calculs sur plusieurs coeurs
                
                st.write("Scores sur les 10 splits utilisés",scores,"\n mean score = ",np.mean(scores))
                st.text("Matrice de confusion : ")
                cm = confusion_matrix(activity_all_test,svc_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(colorbar=False)
                fig, ax = plt.subplots()

            
                cm = confusion_matrix(activity_all_test, svc_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                fig, ax = plt.subplots()
                sns.set_theme(font_scale=1.2)  # Adjust font size
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                st.write(fig)
                st.session_state["Data"] = data
                st.session_state["model"] = svcgrid.best_estimator_
                st.text("Modèle sauvegardé !")

                #------------------


            else: st.text("Vous devez d'abord importer un fichier")




        if bt_vin2:
            if("data" in st.session_state):
                data = st.session_state["data"]
                X=data[data.columns[1:-1]]
                Y=data["target"]

                X = preprocessing.scale(X)
                st.text("Pourcentage jeu d'entrainement : " + str(len(activity_all_train)/ float(len(X))))
                st.text("Pourcentage jeu de test : " + str(len(activity_all_test)/ float(len(X))))

                st.title("Génération du modèle Regression Logistique")
                with st.spinner("Veuillez patienter pendant l'entrainement du modèle"):
                    clf = LogisticRegressionCV(cv=5, random_state=0)

                    #--------------------------------------
                    my_kfold =    model_selection.KFold(n_splits=10, shuffle=True, random_state=0)
                    scores = model_selection.cross_val_score(estimator=clf,
                                            X=X,
                                            y=Y,
                                            cv=my_kfold,
                                            n_jobs=-1) 
                    st.text("Moyenne du score : " + str(mean(scores)))
                    from sklearn.model_selection import cross_val_score, KFold
                st.text("Matrice de confusion : ")
                features_all_train, features_all_test, activity_all_train, activity_all_test = model_selection.train_test_split(X,Y,train_size=0.7,random_state=42)
                clf.fit(features_all_train, activity_all_train)
                predictions_log=clf.predict(features_all_test)
                cm = confusion_matrix(activity_all_test, predictions_log)
                fig, ax = plt.subplots()
                sns.set_theme(font_scale=1.2)  # Adjust font size
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                st.write(fig)
                st.session_state["Data"] = data
                st.session_state["model"] = clf
                st.text("Modèle sauvegardé !")
            else:
                st.text("Vous devez d'abord importer un fichier")
