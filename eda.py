# import Librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

st.set_option('deprecation.showPyplotGlobalUse', False)


def eda(df, target_names, feature_names):
    st.subheader('Exploraty Data Analysis and Visualization')
    st.write("Choose a plot type from the option")
    if st.checkbox("Show raw data"):
        st.write(df)
    # Add option to show/hidle missing value
    if st.checkbox("Show missing value"):
        st.write(df.isna().sum())
    if st.checkbox("Show data Types"):
        st.write(df.dtypes)
    if st.checkbox("Show descriptive Statistics"):
        st.write(df.describe())
    if st.checkbox("Show Correlation Matrix"):
        correlations = df.corr()
        mask = np.triu(np.ones_like(correlations))
        sns.heatmap(correlations, mask=mask, annot=True, cmap="coolwarm")
        st.pyplot()
    if st.checkbox("Show Histogram for each attribues"):
        for col in df.columns:
            fig, ax = plt.subplots()
            ax.hist(df[col], bins=20, density=True, alpha=0.5)
            ax.set_title(col)
            st.pyplot()

    if st.checkbox("Show Density for each attribues"):
        for col in df.columns:
            fig, ax = plt.subplots()
            sns.kdeplot(df[col], fill=True)
            ax.set_title(col)
            st.pyplot(fig)

    if st.checkbox("Show Scatter for each attribues"):
        fig = px.scatter(df, x=feature_names[0], y=feature_names[1])
        st.plotly_chart(fig)
