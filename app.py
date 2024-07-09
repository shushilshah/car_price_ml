import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
model = pickle.load(open('carprice_model.pkl', 'rb'))
st.title("Car Price Prediction System")

data = pd.read_csv('car_data.csv')


def first_word(column_name):
    column_name = column_name.split(' ')[0]
    return column_name.strip()


data['name'] = data['name'].apply(first_word)

name = st.selectbox('Select Car Brand', data['name'].unique())
year = st.slider('Car Manufacture Year', 1994, 2024)
km_driven = st.slider('KM driven', 11, 200000)
fuel = st.selectbox("Fuel Type", data['fuel'].unique())
seller_type = st.selectbox("Seller Type", data['seller_type'].unique())
transmission = st.selectbox("Transmission type", data['transmission'].unique())
owner = st.selectbox("owner", data['owner'].unique())
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider("Maximum Power", 0, 200)
seats = st.slider("Number of Seats", 5, 10)


if st.button("Predict"):
    input = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission,
            owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type',
                 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])
    st.write(input)
    encoder = LabelEncoder()
    input['name'] = encoder.fit_transform(input['name'])
    input['fuel'] = encoder.fit_transform(input['fuel'])
    input['seller_type'] = encoder.fit_transform(input['seller_type'])
    input['transmission'] = encoder.fit_transform(input['transmission'])
    input['owner'] = encoder.fit_transform(input['owner'])

    car_price = model.predict(input)

    st.markdown('Car Price is ' + str(round(car_price[0], 2)))
