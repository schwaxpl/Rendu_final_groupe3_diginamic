import pandas as pd
import numpy as np
import time 
import datetime
import sys
import csv

# Créez un lecteur CSV pour gérer les données
z=0
data=pd.DataFrame(columns=['codcde','villecli','cpcli','qte','timbrecde'])
for line in sys.stdin:

    val=line.split(",")
    l=len(line.split(","))
    act_ind=val[0:(l-1)]
    act_ind.append(val[l-1].strip("\n"))
    if(len(act_ind)==5):
        data.loc[z]=act_ind
        z=z+1
#    csv_reader = csv.reader(line)
#    data=pd.DataFrame(csv_reader)
 
    val=line.split(",")
    l=len(line.split(","))
    act_ind=val[0:(l-1)]#[3].strip("\n")
    act_ind.append(val[l-1].strip("\n"))
    if(len(act_ind)==5):
        data.loc[z]=act_ind
        z=z+1
    

data["qte"]=data["qte"].astype(int)
cmd_sum = data.groupby(["codcde"])["qte"].sum()
s=cmd_sum.sort_values(ascending=False)
df_merge = pd.merge(s, data, on='codcde', how='left')

df_merge=df_merge.drop(["qte_y"],axis=1)
df_merge["qte_commandes"]=df_merge["qte_x"]
df_merge=df_merge.drop(["qte_x"],axis=1)
df_merge=df_merge.drop_duplicates()
df_merge.iloc[0:100,:].to_csv("lot1_result.csv",index=False)
print(df_merge.iloc[0:2,:])

