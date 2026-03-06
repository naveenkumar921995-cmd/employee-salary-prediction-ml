import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl","rb"))

st.title("Employee Salary Prediction")

level = st.slider("Position Level",1.0,10.0,5.0)

prediction = model.predict([[level]])

st.write("Predicted Salary:",prediction[0])
