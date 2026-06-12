import streamlit as st

st.title("Diabetes Detection App")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
age = st.number_input("Age", min_value=0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)

if st.button("Predict"):
    st.success("Prediction goes here")