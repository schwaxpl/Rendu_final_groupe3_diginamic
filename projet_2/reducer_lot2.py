import pandas as pd
import numpy as np
import time 
import datetime
import sys
import csv
import matplotlib
from pylab import *
from io import StringIO

#Lire l'entrée standard en csv
data = pd.read_csv(sys.stdin,engine="python")

#Récupérer les entêtes
dict={}


columns = [ "code_commande", "ville", "qte", "timbre_client", "timbre_commande" ]
#Renommer les entêtes
data.columns = columns
#Convertir la quantité en int
#ça marche pas en pandas 0.18
#data = data.astype({'qte':'int'})
data["qte"]  = pd.to_numeric(data["qte"])#
#Faire un aggrégat de chaque commande avec la moyenne et somme de qte
#ça marche pas en pandas 0.18
#data_by_qte = data.groupby(["code_commande","ville","timbre_client","timbre_commande"]).agg(
#    moy_qte=('qte', "mean"),
#    som_qte=('qte', "sum")).reset_index().sort_values(by=["som_qte",'timbre_commande'],ascending=False)


cmd_sum = data.groupby(["ville"])["qte"].sum()
s=cmd_sum.sort_values(ascending=False).reset_index()
cmd_mean = data.groupby(["ville"])["qte"].mean()
m=cmd_mean.sort_values(ascending=False).reset_index()
df_merge = pd.merge(s,m, on='ville', how='left')
df_merge.rename(index=str,columns={"qte_x":"sum_commandes"},inplace=True)
df_merge.rename(index=str,columns={"qte_y":"moy_commandes"},inplace=True)

print(df_merge)
#On prends le top 100
top_100 = df_merge.head(100)
#Récupération du nombre de lignes / colonnes ( car on veut que le code soit réutilisable sur un DF d'une autre taille )
row,col = top_100.shape
#Calcul du nombre de lignes à récupérer
qte_a_sample = (5/100)*row
#Récupération de N lignes ( sample )
sample = top_100.sample(n=int(floor(qte_a_sample)))

#Export Excel
sample.to_excel("/datavolume1/Final_lot2_groupe3.xlsx")


list_villes = []
list_values = []
#Création de 2 listes avec label / somme d'objets
for ind in sample.index:
    list_villes.append(sample['ville'][ind] + "\n" + str(sample['sum_commandes'][ind]) + " articles" + "\nMoy :" + '{0:.2f}'.format(sample['moy_commandes'][ind]) + "/cde"  )
    list_values.append(sample['sum_commandes'][ind])

#Création du camembert
figure(3, figsize=(4,4))
axes([0.1, 0.1, 0.8, 0.8])
labels=list_villes
fracs=list_values
pie(fracs,labels=labels,textprops={'fontsize': 5})

#Sauvegarde du camembert
savefig('/datavolume1/chart_villes_groupe3.pdf')
