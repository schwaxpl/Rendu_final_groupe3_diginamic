import streamlit as st
import pandas as pd
import Utils.Utils as u
from pylab import *
import matplotlib.pyplot as plt
import random 
from sklearn.decomposition import PCA
from sklearn import preprocessing 
import seaborn as sns
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import statsmodels.formula.api as smf


# Fonction pour afficher le camembert
def afficher_camembert(data):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140)
    ax.set_title('Répartition des valeurs de la variable "sex"')
    ax.axis('equal')
    st.pyplot(fig)

def cross_validation(data, FEATURES, TARGET):
    eps = []
    kf = KFold(n_splits=5, shuffle=True, random_state=2021)
    data_resultat_ridge = []
    split = list(kf.split(data[FEATURES], data[TARGET]))
    for i, (train_index, test_index) in enumerate(split):
        st.write(f'\n ---------------- Fold {i+1} ------------\n')
        st.write(f" -------------- Training on {len(train_index)} samples-------------- ")
        st.write(f" -------------- Validation on {len(test_index)} samples-------------- ")
        
        data_train = data.loc[train_index] 
        data_test = data.loc[test_index]
        # Régression linéaire simple
        clf = Lasso(alpha=0).fit(data_train[FEATURES], data_train[TARGET])
        y_hat = clf.predict(data_test[FEATURES])
        data_resultat_ridge = []
        mse = round(mean_squared_error(data_test[TARGET].values, y_hat), 4)
        res_reg_lin = pd.DataFrame({"variable": FEATURES, "coefficient": clf.coef_, "mse": mse})
        res_reg_lin['model'] = 'Régression linéaire'
        data_resultat_ridge.append(res)
        data_resultat_ridge = pd.concat(data_resultat_ridge)
        st.write(f" MSE Régression Linéaire {np.sqrt(res_reg_lin.mse.min())}")
        st.write(f" MSE Meilleur Ridge {np.sqrt(data_resultat_ridge.mse.min())}")
        st.write(f" MSE Meilleur Lasso {np.sqrt(data_resultat_lasso.mse.min())}")

u.init_page("Extra_aissa")

with st.expander("Traitement pré configuré ""Diabete"" "):
    if("data" in st.session_state):
        data = st.session_state["data"]
        data = data.drop(data.columns[0], axis=1)
    
# Valeurs de la colonne 'sex'
sex_counts = data['sex'].value_counts()


st.title('Camembert pour la variable "sex"')

afficher_camembert(sex_counts)

st.title('Histogrammes de chacune des variables')
for column in data.columns:
    if column != 'target':  
        st.write(f'Histogramme de la variable {column}')
        fig, ax = plt.subplots(figsize=(6, 6))
        data[column].hist(bins=25, ax=ax)
        ax.set_xlabel(column)
        ax.set_ylabel('Fréquence')
        st.pyplot(fig)


# Sélection des variables
FEATURES = [x for x in data.columns if x != 'target']
TARGET = "target"
X = data[FEATURES]
Y = data[TARGET]

lm = smf.ols(formula='target ~ age + sex + bmi + bp + s1 + s2 + s3 + s4 + s5 + s6', data=data).fit()
st.title('Résultats de la régression linéaire')
st.write(lm.summary())

equation = '+'.join(FEATURES)

# Splits
st.title('Validation croisée')
kf = KFold(n_splits=5, shuffle=True, random_state=2000)
split = list(kf.split(data[FEATURES], data[TARGET]))

for i, (train_index, test_index) in enumerate(split):
    st.write(f'\n ---------------- Fold {i+1} ------------\n')
    st.write(f" -------------- Entraînement sur {len(train_index)} échantillons-------------- ")
    st.write(f" -------------- Validation sur {len(test_index)} échantillons-------------- ")
    
    data_train = data.loc[train_index] 
    data_test = data.loc[test_index] 
    lm = smf.ols(formula=f'{TARGET} ~ {equation}', data=data_train).fit()
    y_hat = lm.predict(data_test[FEATURES])
    
    
    r2 = r2_score(data_test[TARGET].values, y_hat)
    mse = mean_squared_error(data_test[TARGET].values, y_hat)
    st.write(f" Fold {i+1} :  MSE {round(mse, 4)}")
    st.write(f" Fold {i+1} :  R2 {round(r2, 4)}")
    
   
    fig, ax = plt.subplots()
    ax.scatter(data_test[TARGET].values, y_hat)
    ax.plot(np.arange(0, 350), np.arange(0, 350), color='red') 
    ax.set_xlabel('Valeurs réelles')
    ax.set_ylabel('Valeurs prédites')
    ax.set_title(f'Fold {i+1}')
    st.pyplot(fig)

n_alphas = 1000
alphas = np.arange(0, n_alphas, 0.1)

st.title('Résultats de la régression Ridge')
data_resultat_ridge = []

for alpha in alphas:
   
    clf = Ridge(alpha=alpha).fit(data[FEATURES], data[TARGET])

    y_hat = clf.predict(data[FEATURES])
    
    mse = round(mean_squared_error(data[TARGET].values, y_hat), 4)
    
    res = pd.DataFrame({"variable": FEATURES, "coefficient": clf.coef_})
    res['alpha'] = alpha
    res['mse'] = mse
    
    data_resultat_ridge.append(res)

data_resultat_ridge = pd.concat(data_resultat_ridge)
st.write(data_resultat_ridge)

n_alphas = 500
alphas = np.arange(0, n_alphas, 0.1)

st.title("Résultats de la regression Lasso")
data_resultat_lasso = []
for alpha in alphas:
    clf = Lasso(alpha=alpha).fit(data[FEATURES], data[TARGET])
    y_hat = clf.predict(data[FEATURES])
    
    mse = round(mean_squared_error(data[TARGET].values, y_hat), 10)
    
    res = pd.DataFrame({"variable": FEATURES, "coefficient": clf.coef_})
    res['alpha'] = alpha
    res['mse'] = mse
    data_resultat_lasso.append(res)
data_resultat_lasso = pd.concat(data_resultat_lasso)
st.write(data_resultat_lasso)

st.title('Comparaison des modèles')
cross_validation(data, FEATURES, TARGET)

