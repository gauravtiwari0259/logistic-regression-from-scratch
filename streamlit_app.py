import streamlit as st
import requests

st.set_page_config(
    page_title="Diabetes Detection App",
    page_icon="🩺",
    layout="centered"
)


st.title("🩺 Diabetes Detection App")
st.write(
    "Enter patient details below and click Predict to check the diabetes risk."
)


pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    value=2
)

glucose = st.number_input(
    "Glucose",
    min_value=0,
    value=120
)

bp = st.number_input(
    "Blood Pressure",
    min_value=0,
    value=70
)

skin = st.number_input(
    "Skin Thickness",
    min_value=0,
    value=20
)

insulin = st.number_input(
    "Insulin",
    min_value=0,
    value=80
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.5
)

age = st.number_input(
    "Age",
    min_value=0,
    value=30
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.5
)

if st.button("🔍 Predict"):

    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "Age": age,
        "DiabetesPedigreeFunction": dpf
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        result = response.json()

        st.divider()

        st.subheader("Prediction Result")

        if result["prediction"] == 1:
            st.error("⚠️ Diabetic")
        else:
            st.success("✅ Non-Diabetic")

        st.metric(
            label="Probability",
            value=f"{result['probability'] * 100:.2f}%"
        )

        st.write("### Model Information")

        st.write(
            f"**Model:** {result['model']}"
        )

        st.write(
            f"**Version:** {result['version']}"
        )

    except Exception as e:

        st.error(
            "Could not connect to FastAPI backend."
        )

        st.write(
            "Make sure FastAPI is running on port 8000."
        )

        st.code(str(e))