#importo le librerie che mi servono
import pandas as pd 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import datetime
import time
from sklearn.linear_model import LinearRegression
import mlflow
from sklearn import preprocessing

#leggo il file csv
data = pd.read_csv('Environmental_original.csv', delimiter = ";")
#leggo i nomi delle colonne contenute nel file
col_names = data.columns.tolist()
#stampo le colonne in modo tale da visualizzarle a video
print("COLUMNS NAME: ")
print(col_names)
#vediamo l'umidità in funzione del tempo (testualmente)
graph1 = data["absolute_humidity"]

#vediamo con un grafico come varia l'umidità in funzione del tempo 
variazione_umidità = graph1.plot()

#i valori di "absolute_humidity" e "srtDate" vengono messi dentro un Array
x = np.array(data['absolute_humidity']).reshape((-1, 1))
y = np.array(data['srtDate'])
print("\nVARIAZIONE DELL'UMIDITA IN FUNZIONE DEL TEMPO: ", x,y)
#*
#*
#*
print("\n\nINIZIO L'ESECUZIONE DI MLFLOW\n")
#inizia l'esecuzione di MLFlow

#sk_model_tree = tree.DecisionTreeClassifier()
#sk_model = sk_model_tree.fit(x, y)
#mlflow.log_model(sk_model, "sk_models")
#sk_model_regression = model.LinearRegression

mlflow.start_run()
mlflow.tracking.MlflowClient()
mlflow.log_param("Parametro", "param")
mlflow.log_metric("Metrica", 100)
mlflow.set_tag('mioTag', 'Valore1')
#mlflow.tracking(uri="/home/maicol/Desktop/Eda Project/EdaProject.py")

#mlflow.end_run()
#uri_artifact = mlflow.get_artifact_uri()
#uri_server = mlflow.set_tracking_uri(uri='/home/maicol/Desktop/Eda Project')
#print("\n\nuri_artifact: ", uri_artifact)
#print("uri_server", uri_server)
#mlflow.create_experiment("EdaProject", artifact_location = 'home/maicol/Desktop/Eda%20Project' )


# LabelEncoder Codifica etichette con valore compreso tra 0 e n_classes-1
le = preprocessing.LabelEncoder()

#Fit label encoder --> indicizza la data con numeri da 0 a n-1
miaData = le.fit_transform(data['srtDate'])
print("\n\nINDICIZZO LA DATA PER NUMERI --> ", miaData)

#Creo un'stanza della classe "Linear Regression" che rappresenterà il modello di regressione
model = LinearRegression()
#con fit() è possibile calcolare valori ottimali per i pesi
pesi = model.fit(x, miaData)
#misuro l'r quadro
r_sq = model.score(x, miaData)
print('\n\nr QUADRO : ', r_sq)

#misuro il punto di intersezione della retta con l'asse y (formula--> y = mx+q)
#intersezione = intercept --> is a scalar
#pendenza = slope --> is an array
print('\nINTERSEZIONE (con asse y): ', model.intercept_)
#misuro la pendenza della retta
print('\nPENDENZA (della retta di regressione):', model.coef_)
#provo a predire valori futuri
y_pred = model.predict(x)
print('\nPREDICTED RESPONSE: ', y_pred, sep = '\n')

sk_model = model
sk_model = sk_model.fit(x, miaData)
#set the artifact_path to location where experiment artifacts will be saved
#log model params
#mlflow.log_param("criterion", sk_model.criterion)
#mlflow.log_param("splitter", sk_model.splitter)
#log model

#mlflow.log_model(sk_model, "sk_models")
print(sk_model)

#print(sk_model)
#print(sk_model_regression)
#plot(sk_model_regression)
mlflow.end_run()
print("\n\nML FLOW TERMINATO\n")