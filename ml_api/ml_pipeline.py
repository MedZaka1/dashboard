import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

iris = load_iris()
X = iris.data
y = iris.target

scalers = [None, StandardScaler(), MinMaxScaler(), Normalizer()]

models = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier(), SVC()]

for scaler in scalers:
    if scaler is not None:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
        
    for model in models:
        model_name = type(model).__name__
        if model_name == 'SVC':
            if scaler is None:
                scaler = Normalizer()
                X_scaled = scaler.fit_transform(X)
            model_name = 'SVC_normalized'
        
        model.fit(X_scaled, y)
        
        filename = f"{model_name}_{type(scaler).__name__}.pickle"
        with open(filename, 'wb') as f:
            pickle.dump((scaler, model), f)