# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:20:24 2020

@author: adamica
"""

import pandas as pd
import numpy as np

#import matplotlib.pyplot as plt
#import seaborn as sns

# Para persistir modelos
import joblib as jb
import pickle

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

def identificar_outliers(df, col):
    import scipy.stats as ss
    iqr = ss.iqr(df[col])
    limite_outlier_superior = np.percentile(df[col], 75) + 1*iqr
    limite_outlier_inferior = np.percentile(df[col], 25) - 1*iqr
    
    not_outlier = [(x <= limite_outlier_superior)|( x >= limite_outlier_inferior) for x in df[col]]
    
    print ("{} tem {} outliers.".format(col, len(not_outlier)-np.sum(not_outlier)))
    return not_outlier

df_raw = pd.read_csv("..\\Bases\\WA_Fn-UseC_-Telco-Customer-Churn.csv")

df_raw['SeniorCitizen'] = ['Yes' if x == 1 else 'No' for x in df_raw.SeniorCitizen]

df_raw['TotalCharges'] = df_raw.TotalCharges.str.replace(" ", "0")
df_raw['TotalCharges'] = pd.to_numeric(df_raw.TotalCharges)



for col in df_raw.columns[df_raw.dtypes!='object']:
    df_raw = df_raw[identificar_outliers(df_raw, col)]
    
    
from sklearn.model_selection import train_test_split

X = df_raw[['gender', 'SeniorCitizen', 'Partner', 'Dependents',
               'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
               'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
               'PaymentMethod', 'MonthlyCharges', 'TotalCharges']]
y = df_raw['Churn']

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(X[X.columns[X.dtypes=='object']])

X_dummy_cat = pd.DataFrame(columns = ohe.get_feature_names(),
                           data= ohe.transform(X[X.columns[X.dtypes=='object']]))

X_dummy = X[X.columns[X.dtypes!='object']].join(X_dummy_cat)

X_train, X_test, y_train, y_test = train_test_split(X_dummy, y, test_size = 0.33)

from sklearn.ensemble import GradientBoostingClassifier

def metrica_de_negocio(y, y_pred, n):
    
    from sklearn.metrics import confusion_matrix
    
    tn, fp, fn, tp = confusion_matrix(y ,y_pred).ravel()
    metrica_negocio = 1 - (fn + n*fp)/(tn+ n*fp + fn + tp)
    
    return metrica_negocio



from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

metrica = make_scorer(metrica_de_negocio, n=2)

model_Boost = GradientBoostingClassifier()
params_Boost = {'learning_rate':(0.05, 0.2),
                'n_estimators':range(20,1000),
                'subsample':[1.0,2.0],
                'min_samples_split':[2,3],
                'max_depth':[2,3,4]
                 }

clf = RandomizedSearchCV(estimator = model_Boost, 
                         param_distributions = params_Boost, 
                         scoring = metrica).fit(X_train, y_train)

model_final = clf.best_estimator_
pickle.dump(model_final, open('..\\Modelos\\modelo.sav', 'wb'))
pickle.dump(ohe, open('..\\Modelos\\encoder.sav', 'wb'))

print('Modelos treinados com sucesso!')


