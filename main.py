
"""
Created on Tue Oct  6 17:30:17 2020

@author: adamica
"""

from flask import Flask, request, jsonify
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import pickle


colunas = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

colunas_cat = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
       
colunas_num = ['tenure', 'MonthlyCharges', 'TotalCharges']


modelo = pickle.load(open('.\\Modelos\\modelo.sav','rb'))
ohe_loaded = pickle.load(open('.\\Modelos\\encoder.sav','rb'))

app = Flask(__name__)

#@app.route('/churn/', methods=['POST'])

@app.route('/', methods=['POST'])
def churn():
    
    dados = request.get_json()
    
    df_input = pd.DataFrame.from_dict(dados, orient='index').T[colunas]
    
    
    X_input = df_input[colunas_num].join(pd.DataFrame(columns = ohe_loaded.get_feature_names(),
                           data= ohe_loaded.transform(df_input[colunas_cat])))

    churn = modelo.predict(X_input)

    return jsonify(churn=churn[0])

app.run(debug=True, port = 5000)