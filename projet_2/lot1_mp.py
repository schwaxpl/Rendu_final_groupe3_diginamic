import pandas as pd
import numpy as np
import time
import datetime
import sys
import csv

# Créez un lecteur CSV pour gérer les données
csv_reader = csv.reader(sys.stdin)
data=pd.DataFrame(list(csv_reader))
dict={}

for i in range(len(data.columns)):
    dict[i]=data.iloc[0,i]


data.rename(columns=dict,inplace=True)
data=data.iloc[1:np.shape(data)[0],:]



#Création d'une colonne année ▒|  partir de la colonne datecde
data['datcde'] = pd.to_datetime(data['datcde'], errors='coerce')
data['annee'] = data['datcde'].dt.year
s=[37524, 37525, 93460, 93461, 93462, 93463, 93761]

# Anticipation des erreurs de la colonne année
data.drop(s)
data['datcde'] = pd.to_datetime(data['datcde'], errors='coerce')
data['annee'] = data['datcde'].dt.year

# Création d'une colonne département ▒|  partir de la colonne cpcli + filtrage entre des années
data["cpcli"] =data.iloc[:,4].astype(str).str[:2]
df_mapper_result = data[(data['annee']>=2006) & (data['annee']<=2010) & data['cpcli'].isin(["53","61","28"])]
#
for ind in df_mapper_result.index:
    print(df_mapper_result.iloc[ind,6],",",df_mapper_result.iloc[ind,5],',',df_mapper_result.iloc[ind,0],',',df_mapper_result.iloc[ind,15],',',df_mapper_result.iloc[ind,9])
