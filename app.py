# import Librairies
from eda import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier  as CART
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

# Import EDA
from eda import eda

st.set_option('deprecation.showPyplotGlobalUse', False)

# load data
dataset = load_iris()

# create dataframe with iris data
data = dataset.data
target_names = dataset.target_names  # classes
feature_names = dataset.feature_names  # columns
target = dataset.target  # Output
df = pd.DataFrame(data, columns=feature_names)

target = pd.Series(target)
# Streamlit
# set up app
st.set_page_config(page_title='EDA and ML Dashboard',
                   layout="centered", initial_sidebar_state="auto")
# Add title and mardown description
st.title("EDA and Predictive Modelling Dashboard")
# define sidebar and sidebar options
option = ["EDA", "Predictive Modelling"]
selected_option = st.sidebar.selectbox("Select an Option ", option)
# EDA

if selected_option == "EDA":
    eda(df, target_names, feature_names)

elif selected_option == "Predictive Modelling":
    st.subheader("Predective Modelling")
    st.write("Choose a transform type and Model")
    X = df.values
    Y = target.values
    test_proportion = .30
    seed = 5
    # Data Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_proportion, random_state=seed)
    # Transform option
    transform_options = ['None', 'StandarScaler', "Normaliser", 'MinMaxScaler']
    transform = st.selectbox("Select Data Transform", transform_options)
    if transform == "StandarScaler":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif transform == "Normaliser":
        scaler = Normalizer()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif transform == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train
    classifier_list = ["LogisticRegression",
                       "SVM",
                       "DescisionTree",
                       "KNeighbors",
                       "RandomForest"]
    classifier = st.selectbox("Select classifier", classifier_list)
    # Add option to select classifiers
    # Add logisticRefression
    if classifier == "LogisticRegression":
        st.write("Here are the results of a LogisticRegression")
        solver_value = st.selectbox("Select solver",
                                    ["lbfgs",
                                     "liblinear",
                                     "newton-cg",
                                     "newton-cholesky"])
        model = LogisticRegression(solver=solver_value)
        model.fit(X_train, Y_train)
        # make prediction
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred, average='micro')
        recall = recall_score(Y_test, y_pred, average='macro')
        f1 = f1_score(Y_test, y_pred, average = "macro")
        confusion_matrix = confusion_matrix(Y_test, y_pred)
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1 Score: {f1}")
        st.write("Confusion matrix: ", confusion_matrix)

    elif classifier == "DescisionTree":
        st.write("Here are the results of a DescisionTree")
        solver_value = st.selectbox("Select solver",
                                    ["lbfgs",
                                     "liblinear",
                                     "newton-cg",
                                     "newton-cholesky"])
        model = CART()
        model.fit(X_train, Y_train)
        # make prediction
        y_pred = model.predict(Y_test.reshape(-1, 1))
        confusion_matrix = confusion_matrix(Y_test, y_pred)
        accuracy = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred, average='micro')
        recall = recall_score(Y_test, y_pred, average='macro')
        f1 = f1_score(Y_test, y_pred, average='micro')
        st.write(f'Accuracy:{accuracy}')
        st.write(f'Predision:{precision}')
        st.write(f'Recall:{recall}')
        st.write(f'FA-Score:{f1}')
        st.write(f'Confusion Matrix:')
        st.write("Confusion matrix: ", confusion_matrix)
        # PREDICTIVE MODELING
