import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import json
import datetime as dt


with open('dt.pkl', 'rb') as f:
    dt_model = pickle.load(f)

with open(r'Category_Columns_Encoded_Data.json', 'r') as f:
    data = json.load(f)

#Streamlit Part 

st.markdown("# <font color='red'>Singapore Resale Flat Prices Predicting</font>", unsafe_allow_html=True)

with st.sidebar:
 select=st.radio("Main Menu",["Home","Select option", "Resale Price"])
    

if select == 'Home':

 col1, col2 = st.columns(2)

 with col1:
    
        col1.header(""":blue[Objective of the Project] : 
The objective of this project is to develop a machine learning 
model and deploy it as a user-friendly web application 
that predicts the resale prices of flats in Singapore. 
This predictive model will be based on historical data 
of resale flat transactions, and it aims to assist both 
potential buyers and sellers in estimating the resale value of a flat.""")
        
        col1.header(""":blue[Project Motivation] : 
The resale flat market in Singapore is highly competitive, 
and it can be challenging to accurately estimate the resale 
value of a flat. There are many factors that can affect 
resale prices, such as location, flat type, floor area, 
and lease duration. A predictive model can help to overcome 
these challenges by providing users with an estimated 
resale price based on these factors.""")
        
        col1.header("""
                    :blue[Technologies used]:
                    1.Python 
                    2.Pandas
                    3.Numpy
                    4.Matplotlib
                    5.Seaborn
                    6.Scikit-learn
                    7.Streamlit
                      """)


 with col2:
        col2.image("Resale.jpg")

 with col1:       
        st.header(":blue[Data Collection and Preprocessing:]")
        st.write("Collected a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB) for the years 1990 to Till Date. Preprocess the data to clean and structure it for machine learning.")

        st.header(":blue[Feature Engineering:]")
        st.write("Extracted relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date. Create any additional features that may enhance prediction accuracy.")

        st.header(":blue[Model Selection and Training:]")
        st.write("Choosen Decission Tree machine learning model for regression and trained the model on the historical data, using a portion of the dataset for training.")
        
 with col2:
        st.header(":blue[Model Evaluation:]")
        st.write("Evaluated the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) and R2 Score.")

        st.header(":blue[Streamlit Web Application:]")
        st.write("Developed a user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey range, etc.). Utilize the trained machine learning model to predict the resale price based on user inputs.")

        st.header(":blue[Deployment on Render:]")
        st.write("Deploy the Streamlit application on the Render platform to make it accessible to users over the internet.")

        st.header(":blue[Testing and Validation:]")
        st.write("Thoroughly tested the deployed application to ensure it functions correctly and provides accurate predictions.")


if select == 'Select option':
   
 col3, col4 = st.columns(2)

 with col3:
    
        col3.header(""":blue[Town Names] : 

    Town_Names = 
    'ANG MO KIO':0,
    'BEDOK':1,
    'BISHAN':2,
    'BUKIT BATOK':3,
    'BUKIT MERAH':4,
    'BUKIT PANJANG':5,
    'BUKIT TIMAH':6,
    'CENTRAL AREA':7,
    'CHOA CHU KANG':8,
    'CLEMENTI':9,
    'GEYLANG':10,
    'HOUGANG':11,
    'JURONG EAST':12,
    'JURONG WEST':13,
    'KALLANG/WHAMPOA':14,
    'LIM CHU KANG':15,
    'MARINE PARADE':16,
    'PASIR RIS':17,
    'PUNGGOL':18,
    'QUEENSTOWN':19,
    'SEMBAWANG':20,
    'SENGKANG':21,
    'SERANGOON':22,
    'TAMPINES':23,
    'TOA PAYOH':24,
    'WOODLANDS':25,
    'YISHUN':26
""")
    
        col4.header(""":blue[Flat Types] :     
    Flat_Types = 
    '1 ROOM': 0,
    '2 ROOM': 1,
    '3 ROOM': 2,
    '4 ROOM': 3,
    '5 ROOM': 4,
    'EXECUTIVE': 5,
    'MULTI-GENERATION': 6
 """)

        
        col4.header(""":blue[Flat Model Types] :     
    Flat_Model_Types = 
    '2-ROOM': 0,
    '3GEN': 1,
    'ADJOINED FLAT': 2,
    'APARTMENT': 3,
    'DBSS': 4,
    'IMPROVED': 5,
    'IMPROVED-MAISONETTE': 6,
    'MAISONETTE': 7,
    'MODEL A': 8,
    'MODEL A-MAISONETTE': 9,
    'MODEL A2': 10,
    'MULTI GENERATION': 11,
    'NEW GENERATION': 12,
    'PREMIUM APARTMENT': 13,
    'PREMIUM APARTMENT LOFT': 14,
    'PREMIUM MAISONETTE': 15,
    'SIMPLIFIED': 16,
    'STANDARD': 17,
    'TERRACE': 18,
    'TYPE S1': 19,
    'TYPE S2': 20 """)
 
if select == 'Resale Price':
    st.markdown("# :green[Predicting Results based on Trained Models]")
    st.header("Predicting Resale Price")

    col1, col2 = st.columns(2, gap='large')

    with col1:
        min_month = 1
        max_month = 12

        # Slider for selecting the month
        selected_month = st.slider("Select the term Month", min_value=min_month, max_value=max_month, value=min_month,
                                step=1)

        town = st.selectbox('Select the **Town**', data['town_before'])

        flat_type = st.selectbox('Select the **Flat Type**', data['flat_type_before'])

        block = st.selectbox('Select the **Block**', data['block_before'])

        street_name = st.selectbox('Select the **Street Name**', data['street_name_before'])
        
    with col2:
    

        flat_model = st.selectbox('Select the **flat_model**', data['flat_model_before'])

        lease_commence_date = st.number_input('Enter the **Lease Commence Year**', min_value=1966, max_value=2022,
                                            value=2017)

        year = st.number_input("Select the transaction Year which you want**", min_value=1990, max_value=2024, value=dt.datetime.now().year)
        
        min_storey = st.number_input('Select the **min_storey**', value=0, min_value=0, max_value=100)

        max_storey = st.number_input('Select the **max_storey**', value=10, min_value=0, max_value=100)

    st.markdown('Click below button to predict the **Flat Resale Price**')
    prediction = st.button('**Predict Price**')

    # Convert categorical values to corresponding encoded values

    town_encoded = data['town_before'].index(town)
    flat_type_encoded = data['flat_type_before'].index(flat_type)
    block_encoded = data['block_before'].index(block)
    street_name_encoded = data['street_name_before'].index(street_name)
    flat_model_encoded = data['flat_model_before'].index(flat_model)

    # Prediction logic
    test_data = [
        selected_month,
        town_encoded,  
        flat_type_encoded, 
        block_encoded,  
        street_name_encoded, 
        flat_model_encoded, 
        lease_commence_date,
        year,
        min_storey,
        max_storey,
    ]

    if prediction:
        # Perform prediction
        test_data_array = np.array([test_data], dtype=np.float32)  # Convert to 2D array with float data type

        predicted_price = dt_model.predict(test_data_array)  # Assuming your model's predict method takes a 2D array

        # Display predicted price
        st.markdown(f"### :blue[Flat Resale Price is] :green[$ {round(predicted_price[0], 3)}]")
