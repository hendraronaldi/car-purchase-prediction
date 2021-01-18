import streamlit as st
import pandas as pd
import numpy as np
import joblib

import tensorflow as tf

@st.cache
def load_dataset():
	return pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')

def load_model():
	return tf.keras.models.load_model('model.h5')

@st.cache
def load_scaler():
	return joblib.load('scaler_x.pkl'), joblib.load('scaler_y.pkl')

@st.cache
def load_label_encoder():
	return joblib.load('encoder.pkl')

def input_features():
	country = st.sidebar.selectbox('Country', countries)
	gender = st.sidebar.selectbox('Gender', (0, 1))
	age = st.sidebar.number_input('Age')
	annual_salary = st.sidebar.number_input('Annual Salary')
	credit_card = st.sidebar.number_input('Credit Card Debt')
	net_worth = st.sidebar.number_input('Net Worth')

	return pd.DataFrame({
		'Country': [country],
		'Gender': [gender],
		'Age': [age],
		'Annual Salary': [annual_salary],
		'Credit Card Debt': [credit_card],
		'Net Worth': [net_worth]
		})

def make_predictions(df):
	pred_df = df.copy()
	pred_df['Country'] = encoder.transform(df['Country'])
	X = scaler_x.transform(pred_df)

	return scaler_y.inverse_transform(model.predict(X))[0][0]


df = load_dataset()
model = load_model()
scaler_x, scaler_y = load_scaler()
encoder = load_label_encoder()
countries = sorted(df.Country.unique())

# sidebar
st.sidebar.header('Input Parameters')
input_df = input_features()

# Main Page
st.title('Car Purchase Prediction')
st.write("""
Detail analysis can be found in notebook: [Link](https://github.com/hendraronaldi/machine_learning_projects/blob/main/MLProject01_CarPurchasePrediction_ANN.ipynb)
""")
st.write("""
Dataset: [Link](https://github.com/hendraronaldi/car-purchase-prediction/blob/master/Car_Purchasing_Data.csv)
""")

st.subheader('Sample Dataset')
st.write(df.head(10))
st.write(df.tail(10))

st.subheader('Input Features')
st.write(input_df)

st.subheader('Car Purchase Amount Prediction')
st.write(make_predictions(input_df))
