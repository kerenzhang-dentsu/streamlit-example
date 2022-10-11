from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np


# Data Config
@st.cache(persist=False,
          allow_output_mutation=True,
          suppress_st_warning=True,
          show_spinner=True)

def get_project_root():
    return str(Path(__file__).parent)

def load_image(name):
    return Image.open(Path(get_project_root()) / f"{name}")

def load_csv():
    df_input = pd.DataFrame()
    df_input = pd.read_csv(input,
                           sep=None,
                           engine='python',
                           encoding='utf-8',
                           parse_dates=True,
                           infer_datetime_format=True)
    return df_input

def prep_data(df,date_col,metric_col):
    df_input = df.rename({date_col:"ds", metric_col:"y"}, errors='raise', axis=1)
    df_input = df_input[['ds', 'y']]
    df_input = df_input.sort_values(by='ds', ascending=True)
    return df_input

def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs(actual - pred) / actual)

# Page Info - Causal Impact Analysis
#st.sidebar.image(load_image('Clearchoice_logo.png'), use_column_width=True)
tabs = ["Lark"]
page = st.sidebar.radio("Applications", tabs)

if page == "Lark":
    st.title('Lark')
    st.markdown("""
                Lark is a python module that gathered 24 different models 
                time series models and machine learning models at ones 
                to make forecast easier. Using Lark, users can save some time 
                and effort by skip the steps building different frameworks 
                and structures for different models which potentially 
                cut the modeling process by over 90%.  
                """)

    st.subheader('1. Data Loading')
    with st.container():
        st.write("Pick a dataset.")
        with st.expander("Data Source"):
            dataset_name = st.selectbox('Select a dataset',
                                        options=['Data Source Name', 'BOA Demo'])
            df = pd.read_csv("BofA_Merill_Edge_Unit_and_Performance_Data_09092022.csv")
            if dataset_name == 'BOA Demo':      
                df


 

