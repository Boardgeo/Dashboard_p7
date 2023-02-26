# Import libraries

import streamlit as st
import shap
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from urllib.request import urlopen
import plotly.express as px
import plotly.offline as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import requests
from PIL import Image

# **********************************************SET THE TITLE PAGE*************************************************************
url_pred = "https://loan-scoring-api2.herokuapp.com/"

st.title("Pret a dépenser - Loan Prediction")
st.subheader("Predict your loan eligibility")
st.markdown("\n")
st.markdown("""---""")
#st.title("Pret a Dépenser")
st.set_option('deprecation.showPyplotGlobalUse', False)

#************************************************GET THE DATASET ***************************************************************#
df_pred = pickle.load(open("models/result_pred.p", "rb"))
df_text = df_pred.drop(['proba', 'prediction'], axis = 1)

# import shap values of the transformed data
model = pickle.load(open("models/Tuned_LGBM_50N.p", "rb"))
Text_encode = pickle.load(open("models/Encoded_shap_ID.p", "rb"))
explainer = pickle.load(open("models/explainer_G.p", "rb"))
shap_values = explainer.shap_values(Text_encode.drop("SK_ID_CURR", axis = 1), check_additivity=False)
explainer1 = pickle.load(open("models/explainer_L.p", "rb"))
shap_values2 = explainer1(Text_encode.drop("SK_ID_CURR", axis = 1), check_additivity=False)

url_pred = "https://loan-scoring-api2.herokuapp.com/"
#response = requests.get(url_pred)
# response.content

#*****************************************************SET VARIABLES************************************************************#
# get list of clients ID
client_list = list(df_pred['SK_ID_CURR'].values)

# set extract client's ID as input search
st.sidebar.markdown("Enter applicant ID")
client_id = st.sidebar.selectbox("Identification number", (client_list))

# client_index number
client_idx = Text_encode[Text_encode['SK_ID_CURR'] == client_id].index[0]

# Get client's data from the ID
client_data = df_pred[df_pred.SK_ID_CURR==int(client_id)]

#*****************************************************SIDE BAR - GET CLIENT DETAILS************************************************#

st.sidebar.write('__Personal Details__')
st.sidebar.write('Gender:', client_data['CODE_GENDER'].values[0])
st.sidebar.write('Age:', round(client_data['DAYS_BIRTH'].values[0]))
st.sidebar.write('Education Level:', client_data['NAME_EDUCATION_TYPE'].values[0])
st.sidebar.write('Occupation:', client_data['OCCUPATION_TYPE'].values[0])
st.sidebar.write('Marital Status:', client_data['NAME_FAMILY_STATUS'].values[0])
st.sidebar.write('Number of Children:', client_data['CNT_CHILDREN'].values[0])


decision_df = df_pred[df_pred["SK_ID_CURR"] == int(client_id)][["proba", "prediction"]]
score = round(decision_df['proba'].iloc[0]*100, 2)

option = st.sidebar.selectbox("__Select an option__", ['Prediction', 'Comparison', 'Interpretation'])
# predict_button = st.sidebar.button("Predict")
    
if option == 'Prediction':
    # Construct the scoring gauge
    st.write("__Customer's Scoring Gauge__")
    st.markdown("\n")
    st.spinner('Scoring gauge loading.......')
    fig = go.Figure()
    fig.add_trace(go.Indicator(
            domain = {'x': [0,1], 'y': [0,1]},
            # client's score in %
            value = score,
            mode = "gauge+number",
            delta = {'reference': 50},
            gauge = {'axis':{'range': [None, 100], 'tickwidth': 3, 'tickcolor': 'darkblue'},
                    'bar': {'color': 'white', 'thickness': 0.15},
                    'bgcolor': 'white', 'borderwidth': 2, 'bordercolor': 'gray',
                    'steps': [{'range': [0, 38], 'color': 'green'},
                                {'range': [38, 42.5], 'color': 'limeGreen'},
                               #{'range': [42.5, 43], 'color': 'red'},
                              {'range': [42.5, 50], 'color': 'orange'},
                             {'range': [50, 100], 'color': 'crimson'}],
                    'threshold': {'line': {'color': 'red', 'width': 5}, 'thickness': 1.0, 'value': 42.4 }}))
    fig.update_layout(paper_bgcolor='white',
                                height=300, width=300,
                                font={'color': 'darkblue', 'family': 'Arial'},
                                margin=dict(l=0, r=0, b=0, t=0, pad=0))
    st.plotly_chart(fig, use_container_width=True)
        

    if score <= 40:
            st.markdown("The applicant has a good potential to repay the loan")
            st.success('Decision: :green[Application Granted]')   

    elif 40 < score <=45:
            st.markdown("Risk of default! The applicant may not repay the loan")
            st.success('Decision: :orange[Application may be granted after providing additional information]') 

    elif 45 < score < 50:
            st.markdown("The applicant may likely not repay the loan")
            st.success('Decision: :orange[Ask for additional infomation or seek approval from your manager]') 
        
    else: 
            st.markdown(":red[Alert!] The applicant has a high risk to not repay the loan!")
            st.error('Decision: :red[Application Refused]')

#*****************************************************COMPARISON WITH OTHER CLIENTS************************************************#
df = df_text.drop('SK_ID_CURR', axis=1)

if option == 'Comparison':

    st.subheader("Comparison with other clients")
    st.markdown("Quantitative features")

    df_pred["Score"] = df_pred['proba']

    # extract few numerical features as list
    num_cols =['Score', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'REG_CITY_NOT_LIVE_CITY',
           'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_REQ_CREDIT_BUREAU_WEEK',
       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
       'AMT_REQ_CREDIT_BUREAU_YEAR', 'CREDIT_INCOME_RATIO', 'PAYMENT_RATE' ]

    num_input = st.selectbox("Select a feature for interactive analysis. Client's position in red:", num_cols)
    st.markdown(num_input)

    x0 = df_pred[df_pred['prediction'] == 0][num_input]
    y0 = df_pred[df_pred['prediction'] == 1][num_input]
    z0 = df_pred[num_input]
    bins = np.linspace(0, 1, 15)

    num_client = df_pred[df_pred["SK_ID_CURR"] == (client_id)][num_input].item()

    group_labels = ['Credit worthy', 'Non-credit worthy','Global']

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x0, name = 'Non defaulters'))
    fig.add_trace(go.Histogram(x=y0, name = 'Defaulters'))
    fig.add_trace(go.Histogram(x=z0,name = 'All the clients' ))
    fig.add_vline(x= num_client, annotation_text = 'client n° '+ str(client_id), line_color = "red")
    fig.update_layout(barmode='relative')
    fig.update_traces(opacity=0.75)
    plt.show()
    st.plotly_chart(fig, use_container_width=True)

#*************************************************Interpretation******************************************************************
if option == 'Interpretation':
    st.subheader("Individual interpretation")
    fig = shap.plots.waterfall(shap_values2[client_idx], show = False)
    st.pyplot(fig)

    # Summary plot of feature components according to their importance
    st.subheader("Global feature interpretation")
    fig = plt.figure()
    fig = shap.summary_plot(shap_values, Text_encode.drop("SK_ID_CURR", axis = 1), 
                        plot_type="bar", max_display=10, show = False)
    st.pyplot(fig)
