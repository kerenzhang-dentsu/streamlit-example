from urllib import response
from click import option
from matplotlib import container

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import holidays
from prophet import Prophet
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from orbit.eda import eda_plot
import orbit.constants.palette as palette
from orbit.utils.plot import get_orbit_style
from orbit.models import DLT
from orbit.diagnostics.plot import plot_predicted_data, plot_predicted_components
from orbit.diagnostics.backtest import BackTester, TimeSeriesSplitter
from orbit.diagnostics.plot import plot_bt_predictions
from orbit.diagnostics.metrics import smape, wmape
from orbit.diagnostics.plot import plot_bt_predictions2
orbit_style = get_orbit_style()
plt.style.use(orbit_style)

# Page Config
st.set_page_config(page_title="Demo",
                   layout="wide")

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

def prep_data(df):
    df_input = df.rename({date_col:"ds", metric_col:"y"}, errors='raise', axis=1)
    df_input = df_input[['ds', 'y']]
    df_input = df_input.sort_values(by='ds', ascending=True)
    return df_input

def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs(actual - pred) / actual)

# Page Info - Causal Impact Analysis
#st.sidebar.image(load_image('Clearchoice_logo.png'), use_column_width=True)
tabs = ["Causal Impact Analysis", "Scenario Planner"]
page = st.sidebar.radio("Applications", tabs)

if page == "Causal Impact Analysis":

    st.title('Causal Impact Analysis')
    st.markdown("""
                dentsu Data Science team uses Causal Impact Analysis to understand 
                the cause-and-effect relationship by looking at leads generation. 
                We predict the baseline numbers using machine learning techniques 
                to remove TV impact to simulaate our expected SEM leads numbers. The 
                difference between actual and baseline is the incremental leads driven 
                by TV. 
                """)
    st.subheader('1. Data Loading')
    with st.container():
        st.write("Pick a dataset.")
        with st.expander("Data Source"):
            dataset_name = st.selectbox('Select a dataset',
                                        options=['Data Source Name', 'SEM Demo'])
            df = pd.read_csv("sem_demo.csv")
            if dataset_name == 'SEM Demo':      
                df
                columns = list(df.columns)
                col1,col2 = st.columns(2)
                with col1:
                    date_col = st.selectbox("Select Data Column", index=0, options=columns, key="date")
                with col2:
                    metric_col = st.selectbox("Select Values Column", index=4, options=columns, key="values")
                df = prep_data(df)
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
            st.write('In this section you can modify the algorithm settings.')
            with st.expander("Horizon"):
                periods_input = st.number_input('Select how many future periods (days) to forecast.',
                min_value=1, max_value=366, value=30)
        
            with st.expander("Seasonality"):
                st.markdown("""The default seasonality used is additive, but the best choice depends on the specific case, 
                therefore specific domain kenowledge is required.""")
                seasonality = st.radio(label='Seasonality',
                                    options=['additive', 'multiplicative'])
            
            with st.expander("Trend Components"):
                st.write("Add or remove components:")
                daily = st.checkbox("Daily")
                weekly = st.checkbox("Weekly")
                monthly = st.checkbox("Monthly")
                yearly = st.checkbox("Yearly")
            
            with st.expander("Growth Model"):
                st.write('Default is a linear growth model.')
                growth = st.radio(label='Growth Model', options=['linear', 'logistic'])
                if growth == 'linear':
                    growth_setting={
                        'cap:':1,
                        'floor':0
                    }
                    cap = 1
                    floor = 1
                    df['cap'] = 1
                    df['floor'] = 1

                if growth == 'logistic':
                    st.info('Configure Saturation')
                    cap = st.slider('Cap', min_value=0.0,max_value=1.0,step=0.05)
                    floor = st.slider('Floor', min_value=0.0,max_value=1.0,step=0.05)
                    if floor > cap:
                        st.error('Invalid settings. Cap must be higher than floor.')
                        growth_setting={}
                    if floor == cap:
                        st.warning('Cap must be higher than floor')
                    else:
                        growth_setting = {
                            'cap':cap,
                            'floor':floor
                        }
                        df['cap'] = cap
                        df['floor'] = floor

            with st.expander('Holidays'):
                countries = ['Country Name', 'United States', 'Italy', 'Spain', 'France', 'Germany']
                with st.container():
                    years=[2021,2022]
                    selected_country = st.selectbox(label='Select Country', options=countries) 
                    if selected_country == 'United States':
                        for date, name in sorted(holidays.US(years=years).items()):
                            st.write(date,name)
                    if selected_country == 'Italy':
                        for date, name in sorted(holidays.IT(years=years).items()):
                            st.write(date,name) 
                    if selected_country == 'Spain':
                        for date, name in sorted(holidays.ES(years=years).items()):
                                st.write(date,name)  
                    if selected_country == 'France':     
                        for date, name in sorted(holidays.FR(years=years).items()):
                                st.write(date,name)
                    if selected_country == 'Germany':
                        for date, name in sorted(holidays.DE(years=years).items()):
                                st.write(date,name)
                    else:
                        holidays = False
                    holidays = st.checkbox('Add country holidays to the model')
            
            with st.expander('Hyperparameters'):
                st.write('In this section it is possible to tune the scaling coefficients.')
                seasonality_scale_values = [0.1, 1.0, 5.0, 10.0]
                changepoint_scale_values = [0.01, 0.1, 0.5, 1.0]
                st.write("The changepoint prior scale determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints.")
                changepoint_scale= st.select_slider(label= 'Changepoint prior scale',options=changepoint_scale_values)
                st.write("The seasonality change point controls the flexibility of the seasonality.")
                seasonality_scale= st.select_slider(label= 'Seasonality prior scale',options=seasonality_scale_values)

        submitted = st.form_submit_button("Submit")
    
    if submitted:
        st.markdown(f"""Model Configuration: \n
Horizon: {periods_input} days \n
Seasonality: {seasonality} \n
Trend Components: {daily};{weekly};{monthly};{yearly} \n
Growth: {growth} \n
Holidays: {selected_country} \n
Hyperparameters: changepoints {changepoint_scale}, seasonality {seasonality_scale}
""")
        st.success("Configuration Submitted")
        st.write(df.head())

    st.write("Below you can upload actual SEM and TV to compare with forecast")
    with st.expander("Actual Data"):
        actualdata_input = st.file_uploader(
            'Upload SEM and TV time series of values to validate the impact.'
        )
        if actualdata_input:
            actual_df = pd.read_csv(actualdata_input)
            actual_df
    
    with st.container():
        st.subheader("3. Forecast")
        st.write("Fit the model on the data and generate future prediction.")
        st.write("Load a time series to activate.")
        if dataset_name == 'SEM Demo':
            if st.checkbox("Initialize Model (Fit)", key="fit"):
                m = Prophet(seasonality_mode=seasonality,
                            daily_seasonality=daily,
                            weekly_seasonality=weekly,
                            yearly_seasonality=yearly,
                            growth=growth,
                            changepoint_prior_scale=changepoint_scale,
                            seasonality_prior_scale=seasonality_scale
                            )
                if holidays:
                    m.add_country_holidays(country_name='US')
                if monthly:
                    m.add_seasonality(name='monthly', period=30.4375, fourier_order=5)
                with st.spinner('Fitting the model...'):
                    m = m.fit(df)
                    future = m.make_future_dataframe(periods=periods_input,freq='D')
                    future['cap']=cap
                    future['floor']=floor
                    st.write("The model will froduce forecast up to ", future['ds'].max())
                    st.success('Model fitted successfully')    

            if st.checkbox("Generate Forecast (Predict)", key="predict"):
                try:
                    with st.spinner("Forecasting.."):
                        forecast = m.predict(future)
                        st.success('Prediction generated successfully')
                        forecast = forecast[['ds', 'yhat']]
                        forecast['ds'] = forecast['ds'].dt.date
                        df['ds'] = pd.to_datetime(df['ds']).dt.date
                        eval_df = df[['ds', 'y']].merge(forecast,how='left', on='ds')
                        col1,col2,col3 = st.columns(3)
                        col1.markdown(
                            f"<p style='color: #ff0066; "
                            f"font-weight: bold; font-size: 20px;'> Actual</p>",
                            unsafe_allow_html=True
                        )
                        col1.write('{:,.0f}'.format(eval_df['y'].sum()))
                        col2.markdown(
                            f"<p style='color: #ff0066; "
                            f"font-weight: bold; font-size: 20px;'> Prediction</p>",
                            unsafe_allow_html=True
                        )
                        col2.write('{:,.0f}'.format(eval_df['yhat'].sum()))
                        col3.markdown(
                            f"<p style='color: #ff0066; "
                            f"font-weight: bold; font-size: 20px;'> MAPE</p>",
                            unsafe_allow_html=True
                        )
                        col3.write("{:.2%}".format(mape(eval_df['y'], eval_df['yhat']),2))

                        fig = px.line(eval_df,
                                      x='ds',
                                      y=['y', 'yhat'],
                                      color_discrete_sequence=["#ff0066", "#0477BF", "#ff9933", "#337788","#429e79", "#474747", "#f7d126", "#ee5eab", "#b8b8b8"],
                                      hover_data={"variable":True, "value":":.4f","ds":False})
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
                            legend_title_text="",
                            height=500,
                            width=800,
                            title_text="Actual vs Prediction",
                            title_x=0.5,
                            title_y=1,
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        output = 1

                except:
                    st.warning("You need to train the model first..")

    with st.container():
        st.subheader("4. Causal Impact")
        st.write("Generate causal impact from upload actual file.")
        if actualdata_input:
            if st.checkbox("Generate Causal Impact Summary", key="summary"):
                 with st.spinner("Generating.."):
                    actual_df = prep_data(actual_df)
                    actual_df['ds'] = pd.to_datetime(actual_df['ds']).dt.date
                    ci_data = actual_df.merge(forecast, how='left', on='ds')
                    col1,col2,col3 = st.columns(3)
                    col1.markdown(
                        f"<p style='color: #ff0066; "
                        f"font-weight: bold; font-size: 20px;'> Baseline</p>",
                        unsafe_allow_html=True
                    )
                    col1.write('{:,.0f}'.format(ci_data['yhat'].sum()))
                    col2.markdown(
                        f"<p style='color: #ff0066; "
                        f"font-weight: bold; font-size: 20px;'> Actual</p>",
                        unsafe_allow_html=True
                    )
                    col2.write('{:,.0f}'.format(ci_data['y'].sum()))
                    col3.markdown(
                        f"<p style='color: #ff0066; "
                        f"font-weight: bold; font-size: 20px;'> Impact</p>",
                        unsafe_allow_html=True
                    )
                    col3.write("{:.2%}".format((ci_data['y'].sum() - ci_data['yhat'].sum()) / ci_data['yhat'].sum()))
                    ci_data_plot = eval_df.append(ci_data, ignore_index=True)
                    fig = px.line(ci_data_plot,
                                      x='ds',
                                      y=['y', 'yhat'],
                                      color_discrete_sequence=["#ff0066", "#0477BF", "#ff9933", "#337788","#429e79", "#474747", "#f7d126", "#ee5eab", "#b8b8b8"],
                                      hover_data={"variable":True, "value":":.4f","ds":False})
                    fig.update_xaxes(
                        rangeslider_visible=True,
                        range = ['2022-04-01', '2022-04-27'],
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
                        legend_title_text="",
                        height=500,
                        width=800,
                        title_text="Actual vs Baseline",
                        title_x=0.5,
                        title_y=1,
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    output = 1

if page == "Scenario Planner":
    st.title('Scenario Planner')
    st.markdown("""
                dentsu Data Science team establishs Bayesian Time Series Modeling and Inference to understand 
                the channel contribution and media effectiveness. In addition, we leverages probabilistic program 
                to simulate and forecast different scenario to support multiple investment and optimization decisions.
                """)
    st.subheader('1. Data Loading')
    with st.container():
        st.write("Pick a dataset.")
        with st.expander("Data Source"):
            dataset_name = st.selectbox('Select a dataset',
                                        options=['Data Source Name', 'MEM Demo'])
            df = pd.read_csv("mem_demo.csv")
            if dataset_name == 'MEM Demo':      
                df['Date'] = pd.to_datetime(df['Date'])
                df
                st.pyplot(eda_plot.wrap_plot_ts(df,
                                               date_col='Date',
                                               var_list=df.columns)
                                               )
                df['SEM_Spend_Log'] = np.log(df['SEM_Spend'])
                df['SOCIAL_Spend_Log'] = np.log(df['SOCIAL_Spend'])
                df['TV_Spend_Log'] = np.log(df['TV_Spend'])
                df['TOTAL_Leads_Log'] = np.log(df['TOTAL_Leads'])
    
    st.subheader("2. Parameters Configuration")
    with st.form("config"):
        with st.container():
            st.write('In this section you can modify the algorithm settings.')
            with st.expander("Test Period"):
                periods_input = st.number_input('Select how many test periods (days) to validate.',
                min_value=1, max_value=366, value=30)
            with st.expander("Estimator"):
                st.markdown("""The default estimator used is stan-map, but the best choice depends on the specific case.""")
                estimator = st.radio(label='Estimator',
                                    options=['stan-map', 'stan-mcmc'])
            with st.expander("Seasonality"):
                seasonality_input = st.number_input('Select seasonality frequency number.',
                min_value=4, max_value=365, value=365)
            with st.expander("Regressors"):
                st.write("Add or remove regressors:")
                regressors_positive = st.multiselect('Select regressors you want to add as positive impact.',
                                            df.columns)
                regressors_negative = st.multiselect('Select regressors you want to add as negative impact.',
                                            df.columns)                         
                regressors_notsure = st.multiselect('Select regressors you want to add if you are not sure about the impact.',
                                            df.columns)
            with st.expander("Global Trend"):
                st.markdown("""The default is linear model.""")
                global_trend_option = st.radio(label='Global Trend Option',
                                    options=['linear', 'loglinear', 'flat', 'logistic'])
        submitted = st.form_submit_button("Submit")
    if submitted:
        st.markdown(f"""Model Configuration: \n
Test Period: {periods_input} days \n
Estimator: {estimator} \n
Seasonality: {seasonality_input} \n
Regressors: + {regressors_positive}; - {regressors_negative}; = {regressors_notsure} \n
Global Trend: {global_trend_option} 
""")
        st.success("Configuration Submitted")
    
    with st.container():
        st.subheader("3. Train The Model")
        st.write("Fit the model on the data and generate insights.")
        st.write("Load a dataset to activate.")
        if dataset_name == 'MEM Demo':
            if st.checkbox("Initialize Model (Fit)", key="fit"):
                with st.spinner("Training.."): 
                    train_df = df[:-int(periods_input)]
                    test_df = df[-int(periods_input):]
                    regressor_col=[]
                    regressor_sign=[]
                    if regressors_positive:
                        for i in regressors_positive:
                            regressor_col.append(i)
                            regressor_sign.append("+")
                    if regressors_negative:
                        for i in regressors_negative:
                            regressor_col.append(i)
                            regressor_sign.append("-")     
                    if regressors_notsure:
                        for i in regressors_notsure:
                            regressor_col.append(i)
                            regressor_sign.append("=")    
                    dlt_model = DLT(response_col = 'TOTAL_Leads_Log',
                                    date_col = 'Date',
                                    estimator = estimator,
                                    seasonality = int(seasonality_input),
                                    regressor_col = regressor_col,
                                    regressor_sign = regressor_sign,
                                    global_trend_option = global_trend_option)
                    dlt_model.fit(train_df)
                    predicted_df = dlt_model.predict(test_df, decompose=True)
                    regression_coefs = dlt_model.get_regression_coefs()
                    col1,col2,col3 = st.columns(3)
                    col1.markdown(
                        f"<p style='color: #ff0066; "
                        f"font-weight: bold; font-size: 20px;'> Actual</p>",
                        unsafe_allow_html=True
                    )
                    col1.write('{:,.0f}'.format(test_df['TOTAL_Leads'].sum()))
                    col2.markdown(
                        f"<p style='color: #ff0066; "
                        f"font-weight: bold; font-size: 20px;'> Prediction</p>",
                        unsafe_allow_html=True
                    )
                    col2.write('{:,.0f}'.format(np.exp(predicted_df['prediction']).sum()))
                    col3.markdown(
                        f"<p style='color: #ff0066; "
                        f"font-weight: bold; font-size: 20px;'> MAPE</p>",
                        unsafe_allow_html=True
                    )
                    col3.write("{:.2%}".format(mape(test_df['TOTAL_Leads'], np.exp(predicted_df['prediction'])),2))
                    fig = go.Figure([go.Bar(x=regression_coefs['regressor'], y=regression_coefs['coefficient'])])
                    fig.update_traces(marker_color=["#0477BF", "#ff9933", "#337788"], marker_line_color="#ff0066",marker_line_width=1.5, opacity=0.6)
                    fig.update_layout(title_text='Channel Contribution')
                    st.plotly_chart(fig, use_container_width=True)
                    output = 0
    
    with st.container():
        st.subheader("4. Scenario Planning")
        st.write("Generate future forecast from customlized scenario.")
        with st.form("scenario"):
            with st.container():
                with st.expander("Scenario Plan"):
                    scenario_periods_input = st.number_input('Select how many future periods (days) to forecast.',
                                                            min_value=1, max_value=365, value=30)
                    sem_budget = st.number_input('Select how much budget for SEM.',
                                                min_value=1, max_value=100000000, value=1)
                    social_budget = st.number_input('Select how much budget for SOCIAL.',
                                                min_value=1, max_value=100000000, value=1)
                    tv_budget = st.number_input('Select how much budget for TV.',
                                                min_value=1, max_value=100000000, value=1)
               
            scenario_submitted = st.form_submit_button("Submit")
        if scenario_submitted:
            st.markdown(f"""Scenario Submission: \n
Scenario Period: {scenario_periods_input} days \n
SEM Budget: {sem_budget} \n
Social Budeget: {social_budget} \n
TV Budget: {tv_budget} 
""")
            st.success("Scenario Submitted")
            future_df = dlt_model.make_future_df(periods=int(scenario_periods_input))
            future_df['SEM_Spend_Log'] = np.log(sem_budget/scenario_periods_input)
            future_df['SOCIAL_Spend_Log'] = np.log(social_budget/scenario_periods_input)
            future_df['TV_Spend_Log'] = np.log(tv_budget/scenario_periods_input)
            scenario_predicted_df = dlt_model.predict(future_df)
            st.markdown(
                        f"<p style='color: #ff0066; "
                        f"font-weight: bold; font-size: 20px;'> Prediction</p>",
                        unsafe_allow_html=True
                        )
            st.write('{:,.0f}'.format(np.exp(scenario_predicted_df['prediction']).sum()))
            scenario_predicted_df['TOTAL_Leads'] = np.exp(scenario_predicted_df['prediction'])
            scenario_predicted_df = scenario_predicted_df[['Date', 'TOTAL_Leads']]
            scenario_df_plot = train_df[['Date', 'TOTAL_Leads']].append(scenario_predicted_df)
            fig = px.line(scenario_df_plot,
                        x='Date',
                        y='TOTAL_Leads',
                        color_discrete_sequence=["#ff0066", "#0477BF", "#ff9933", "#337788","#429e79", "#474747", "#f7d126", "#ee5eab", "#b8b8b8"])
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
                yaxis_title="Total Leads",
                legend_title_text="",
                height=500,
                width=800,
                title_text="Scenario Prediction",
                title_x=0.5,
                title_y=1,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
            output = 1




                        
                
               
