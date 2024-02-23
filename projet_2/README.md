# Projet_datafromage_groupe3

Problématique :

On cherche à analyser les données d'une fromagerie sur 3 axes :

Lot 1 :
Ressortir les 100 meilleures commandes de 2006 à 2010 dans les départements 53, 61 et 28
Format : Excel

Lot 2 :
Ressortir les 5 ( 100 x 5% ) meilleures commandes de 2011 à 2016 dans les départements 22, 49 et 53
avec la moyenne des quantités commandées pour chaque article
Format : Excel

Sortir un graph "camembert" des villes avec le plus de commandes
Format : PDF
Lot 3 :
Intégrer des les données du datawarehouse dans PowerBI via une base HBase

# Procédure lancement mapreduce
Importer les datas et les scripts pythons via filezilla sur la vm linux

utiliser la commande docker cp pour échanger des fichiers entre la vm et le docker hadoop-master:
- docker cp /repertoire/initial/monfichier1 /repertoire/destination/

envoyer le csv sur hdfs via hadoop fs copyfromlocal

lancer la commande hadoop streaming en lui donnant le mapper et reducer en paramètre

hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.7.2.jar -file /root/mapper_lot2.py -mapper "python3 mapper_lot2.py" -file /root/reducer_lot2.py -reducer "python3 reducer_lot2.py" -input /user/root/datafromage.csv -output /user/root/sortie

# Procédure import données
Prérequis :
Putty
Filezilla
 
## Dans Filezilla, se connecter à la machine, même ports/host/user que pour putty
 
mettre le csv depuis la machine locale
 
## Dans putty
 
## Démarrer les dockers master & slave
./start_docker_digi.sh

./lance_srv_slaves.sh
## Copier le csv depuis la VM vers le docker
docker cp ./dataw_fro03.csv hadoop-master:/root/datafromage.csv

## Rentrer sur le docker hadoop master
./bash_hadoop_master.sh

## Initialiser la machine
./start-hadoop.sh

./happybase.sh

./hbase_odbc_rest.sh

## Créer la table sur hbase
hbase shell

create 'data_fromage',{NAME=>'cf'}

exit

## Préparer son fichier pour l'import
Script pour ajouter le numéro de ligne au début de chaque ligne du csv

awk -v OFS=',' '{print NR,$0}' /root/datafromage.csv > /root/datafromage_prepare.csv

## Envoyer le fichier sur le hdfs
hadoop fs -copyFromLocal /root/datafromage_prepare.csv /user/root/

## Importer son fichier sur hbase
hbase org.apache.hadoop.hbase.mapreduce.ImportTsv -Dimporttsv.separator=',' -Dimporttsv.columns='HBASE_ROW_KEY,cf:codcli,cf:genrecli,cf:nomcli,cf:prenomcli,cf:cpcli,cf:villecli,cf:codcde,cf:datcde,cf:timbrecli,cf:timbrecde,cf:Nbcolis,cf:cheqcli,cf:barchive,cf:bstock,cf:codobj,cf:qte,cf:Colis,cf:libobj,cf:Tailleobj,cf:Poidsobj,cf:points,cf:indispobj,cf:libcondit,cf:prixcond,cf:puobj' data_fromage /user/root/datafromage_prepare.csv

## Configurer son driver odbc hbase
Ajouter le bon hostname / port en fonction de votre serveur

Et voilà, le driver ODBC est maintenant utilisable par PowerBI !

( fichier de nettoyage et structuration des données PBI fourni )


