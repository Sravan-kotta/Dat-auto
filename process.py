import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pycaret.classification import setup, compare_models, pull, save_model
import ipywidgets
import plotly.graph_objects as go
@st.cache_data
def preprocess_pca(data,target):
    clf1 = setup(data = data, target = target, pca = True, pca_components = 2, normalize = True)
    # Standardize features
    pca_df = clf1.dataset_transformed
    return pca_df
@st.cache_data
def isolation_forest_visual(pca,w):
    # Standardize features
    isf = IsolationForest(contamination=w)
    isf.fit(pca)
    predictions = isf.predict(pca)
    pca["iso_forest_outliers"] = predictions
    
    colors = {1: 'b', -1: 'r'}
    plt.clf()
    fig = plt.figure()
    plt.scatter(pca['pca0'], pca['pca1'], color=[colors[r] for r in pca['iso_forest_outliers']])
    st.pyplot(fig)
    
def iforest_visual():
    fig = ipywidgets.interact(isolation_forest_visual,w=(0.001, 0.5, 0.01))
    st.ploty_chart(fig)
    