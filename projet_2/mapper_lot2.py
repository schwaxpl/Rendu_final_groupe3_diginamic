#!/usr/bin/env python
"""mapper.py"""

min_year = 2011
max_year = 2016
import sys
cpt = 0
# input comes from STDIN (standard input)
texte_csv = []
for line in sys.stdin:
    ligne_bonne = True
    # remove leading and trailing whitespace
    line = line.strip() # il supprime les espaces devant et derrière
    # 6 colonnes de valeurs séparées par une tabulation
    # date heure magasin produit cout paiement
    commande = line.replace('"','').split(',')
    if len(commande) != 25:
        continue # je ne fais pas le code en dessous
    # Je récupère une liste de string
    codcde = commande[6]
    villecli = commande[5]
    qte = commande[15]
    timbrecli = commande[8]
    timbrecde = commande[9]
    date = commande[7]
    year = date[:4]
    if(year.isdigit()):
        if(not ((min_year <= int(year)) and (int(year) <= max_year))):
            ligne_bonne = False
    else:
        ligne_bonne = False
        
    if(timbrecli != "0" and timbrecli != "null"):
        ligne_bonne = False
    
    if("NULL" in qte):
        ligne_bonne = False


    if(ligne_bonne and cpt>0):
        texte_csv.append( codcde+','+villecli+','+qte+','+timbrecli+','+timbrecde)
    cpt += 1
for l in texte_csv:
    print(l)