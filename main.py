import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model
from pandas.api.types import is_numeric_dtype

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload Your Dataset", "Exploratory Data Analysis", "Data Preprocessing", "Machine Learning", "Download Trained Model"])
    st.info("This project application helps you to Automate Machine Learning and Data Analysis ")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload Your Dataset":
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    profile_report = ProfileReport(df)
    st_profile_report(profile_report)

if choice == "Data Preprocessing":
    st.title("Data Preprocessing")
    target = st.selectbox("Select Your Target", df.columns)
    exec(open(r"C:\Users\srava\Desktop\DATAUTO\Dat-auto\process.py").read())
    choice = st.radio(r"$\textsf{\Large Navigation}$",["Getting started", "Outlier removal"])#, "Exploratory Data Analysis", "Data Preprocessing", "Machine Learning", "Download Trained Model"])
    if choice == "Getting started":
        label = r'''$\textsf{ \scriptsize Visualizing hyperparameter selection }$'''
        st.title(label)
        st.write("Dont forget to select target :smile:")
    if choice == "Outlier removal":  
        pca = preprocess_pca(df, target)
        st.title(r"$\textbf{ \scriptsize Using Isolation Forest}$")
        w = st.slider(r"$\textsf{\Large Enter contamination}$",0.01,0.5,0.01)
        isolation_forest_visual(pca,w)
            
    

    
    


if choice == "Machine Learning":
    st.title("All Machine Learning Models")
    target = st.selectbox("Select Your Target", df.columns)
    
    # Check and handle non-numeric values in the target column
    if not is_numeric_dtype(df[target]):
        st.error(f"Error: Target column '{target}' is not numeric. Please handle non-numeric values.")
    else:
        if st.button("Train Model"):
            try:
                setup(df, target=target,fix_imbalance = True)
                setup_df = pull()
                st.info("This is the ML Experiment settings")
                st.dataframe(setup_df)
                
                best_model = compare_models()
                compare_df = pull()
                st.info("This is the ML Model")
                st.dataframe(compare_df)
                
                save_model(best_model, 'best_model')
                
            except Exception as e:
                st.error(f"An error occurred during model training: {str(e)}")

if choice == "Download Trained Model":
    try:
        with open("best_model.pkl", 'rb') as f:
            st.download_button("Download the Model", f.read(), "trained_model.pkl")
    except FileNotFoundError:
        st.warning("Model file 'best_model.pkl' not found. Please train a model first.")
