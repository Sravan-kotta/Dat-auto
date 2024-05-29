import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model
from pandas.api.types import is_numeric_dtype


def preprocess_data(df):
  
  # Handle missing values (you can customize this further)
  # Option 1: Remove rows with missing values (be cautious of data loss)
  # df.dropna(inplace=True)

  # Option 2: Impute missing values (e.g., with mean/median/mode)
  for col in df.columns:
    if df[col].isnull().any():  # Use isnull directly on Series
      if pd.api.types.is_numeric_dtype(df[col]):
        df[col].fillna(df[col].mean(), inplace=True)
      else:
        df[col].fillna(df[col].mode()[0], inplace=True)

  # Handle redundancy (you can add more checks here)
  # Identify and remove duplicate rows
  df = df.drop_duplicates()

  return df


with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Dat-Auto")
    choice = st.radio("Navigation", [
        "Upload Your Dataset",
        "Preprocess Data",
        "Exploratory Data Analysis",
        "Machine Learning",
        "Download Trained Model"
    ])
    st.info("This project application helps you to Automate Machine Learning and Data Analysis ")
    st.info("The Preprocess section handles with Missing Values, Redundancy and other elements that may be unnecessary in the dataset")
    st.info("Exploratory Data Analysis (EDA) Gives you a summary of the entire dataset involving all of its contents")
    st.info("The Machine Learning stage trains all ML Models on your dataset and returns the output of all of them giving you to download the best performing model")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload Your Dataset":
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Preprocess Data":
    st.title("Data Preprocessing")
    if st.button("Preprocess Data"):
        if os.path.exists("sourcedata.csv"):
            df = pd.read_csv("sourcedata.csv", index_col=None)
            df_preprocessed = preprocess_data(df.copy())  # Avoid modifying original data
            st.success("Data preprocessing complete!")
            st.dataframe(df_preprocessed)
        else:
            st.warning("Please upload a dataset first.")

if choice == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    # Use the preprocessed data if available
    if "df_preprocessed" in locals():
        profile_report = ProfileReport(df_preprocessed)
    else:
        profile_report = ProfileReport(df)
    st_profile_report(profile_report)

if choice == "Machine Learning":
    st.title("All Machine Learning Models")
    target = st.selectbox("Select Your Target", df.columns)
    
    # Check and handle non-numeric values in the target column
    if not is_numeric_dtype(df[target]):
        st.error(f"Error: Target column '{target}' is not numeric. Please handle non-numeric values.")
    else:
        if st.button("Train Model"):
            try:
                setup(df, target=target)
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