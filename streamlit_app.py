from collections import namedtuple
from select import select
import altair as alt
import math
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import plotly.express as px
import subprocess
import sys
#manually install xlrd and lark because dockerfile cannot install them somehow
# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install('git+https://github.com/dentsu-Media-US-Data-Science/Lark.git')


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
                st.write('Corrlations:')
                st.dataframe(df.corr())
                cols = ['Date','Clicks', 'Cost', 'Impressions', 'Accounts_Opened', 'Funded_Units']
                columns = cols
                col1,col2 = st.columns(2)
                with col1:
                    date_col = st.selectbox("Select Data Column", index=0, options=columns, key="date")
                with col2:
                    metric_col = st.selectbox("Select Values Column", index=4, options=columns, key="values")

                df = prep_data(df,date_col,metric_col)
                output = 0
                fig = px.line(df,
                              x='ds',
                              y='y',
                              color_discrete_sequence=["#ff0066", "#0477BF", "#ff9933", "#337788","#429e79", "#474747", "#f7d126", "#ee5eab", "#b8b8b8"]
                              )
                fig.update_xaxes(
                            rangeslider_visible=True,
                            rangeselector=dict(
                                buttons=list(
                                    [
                                        dict(count=7, label="1w", step="day", stepmode="backward"),
                                        dict(count=1, label="1m", step="month", stepmode="backward"),
                                        dict(count=3, label="3m", step="month", stepmode="backward"),
                                        dict(count=6, label="6m", step="month", stepmode="backward"),
                                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                                        dict(count=1, label="1y", step="year", stepmode="backward"),
                                        dict(step="all"),
                                    ]
                                )
                            ),
                        )
                fig.update_layout(
                            yaxis_title=metric_col,
                            xaxis_title=date_col,
                            legend_title_text="",
                            height=500,
                            width=800,
                            title_text="Time Series Preview",
                            title_x=0.5,
                            title_y=1,
                            hovermode="x unified",
                        )
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("2. Parameters Configuration")
    with st.form("config"):
        with st.container():
            st.write('In this section you can choose the model you want to use.')
            with st.expander("Models"):
                model_list = [
                                'Arima', 
                                'Prophet', 
                                'Random Forest', 
                                'Extreme Gradient Boosting', 
                                'Gradient Boosting', 
                                'Cat Boost',
                                'Light Gradient Boosting Machine',
                                'Extra Trees',
                                'Ada Boost',
                                'Ridge Regression',
                                'Bayesian Ridge Regression',
                                'Lasso Regression',
                                'Theil-Sen Regression',
                                'Least Angle Regression',
                                'Lasso Least Angle Regression',
                                'Decision Tree',
                                'Huber Regressor',
                                'Random Sample Consensus',
                                'Orthogonal Matching Pursuit',
                                'Passive Aggressive Regressor',
                                'Elastic Net',
                                'K-Neighbors Regression',
                                'Support Vector Machine'
                                ]
                selected_model = st.selectbox(label='Select model', options=model_list)
            with st.expander("Parameters:"):
                st.write('In this section it is possible to modify pamameters of your model.')
                if selected_model == 'Arima':
                    p = [1,2,3,4]
                    d = [1,2,3,4]
                    q = [1,2,3,4]
                    p_scale = st.select_slider(label = 'P scale', options = p)
                    d_scale = st.select_slider(label = 'D scale', options = d)
                    q_scale = st.select_slider(label = 'Q scale', options = q)
                elif selected_model == 'Prophet':
                    weekly_seasonality = [*range(1,53,1)]
                    yearly_seasonality = [*range(1,366,1)]
                    seasonality_prior_scale = np.arange(0.0,11.0,0.1)
                    changepoint_range = [*range(1,10,1)]
                    weekly_scale = st.select_slider(label = 'Weekly seasonality scale', options = weekly_seasonality)
                    yearly_scale = st.select_slider(label = 'Yearly seasonality scale', options = yearly_seasonality)
                    seasonality_scale = st.select_slider(label = 'Seasonality prior scale', options = seasonality_prior_scale)
                    changepoint_scale = st.select_slider(label = 'Change point scale', options = changepoint_range)
                elif selected_model == 'Random Forest':
                    criterion = ['absolute_error', 'squared_error','poisson']
                    criterion_selection = st.selectbox(label = 'Criterion selection', options = criterion)
                else:
                    st.write('Pamaters have not been built for this model yet, check back later!')
            with st.expander("Forecast days"):
                st.write("In this section it is possible to select number of days to forecast.")
                future = st.number_input(label = 'Enter the number of days to forecast')
            with st.expander('Including history'):
                st.write("In this section it is possible to include history or not.")
                history = st.selectbox(label = 'Choose whether to include history or not', options = [True, False])
            with st.expander('With cost'):
                st.write("In this section it is possible to decide whether to use cost as regressor.")
                with_cost = st.selectbox(label = 'Choose whether to use cost as a regressor', options = [True, False])
        submitted = st.form_submit_button("Submit")


    if submitted:
        st.markdown(f"""
                    Model Configuration: \n
                    Model: {selected_model} \n
                    Pamaters: Valid! \n
                    Forecast days: {future} \n
                    Including history: {history} \n
                    With cost: {with_cost} \n
                    """)
        st.success("Configuration Submitted")

    if with_cost:
        st.write("Below you can upload a dataframe with future cost.")
        with st.expander("Future cost"):
            actualdata_input = st.file_uploader(
                'Upload a dataframe with future cost and date.'
            )
            if actualdata_input:
                actual_df = pd.read_csv(actualdata_input)
                actual_df
                columns = list(actual_df.columns)
                col3,col4 = st.columns(2)
                with col3:
                    date_col_cost = st.selectbox("Select Data Column", index=0, options=columns, key="date")
                with col4:
                    cost_col = st.selectbox("Select Cost Column", index=4, options=columns, key="values")
    
    with st.container():
        st.subheader("3. Forecast")
        st.write("Fit the model on the data and generate future prediction.")