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

df = load_dataset()
model = load_model()
scaler_x, scaler_y = load_scaler()
encoder = load_label_encoder()