from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import pickle

app = FastAPI(
    title="Diabetes Detection API",
    description="""
    Predict whether a patient is diabetic using a custom Logistic Regression
    model implemented from scratch using NumPy.

    Features:
    - Custom Gradient Descent
    - L2 Regularization
    - Feature Scaling
    - FastAPI Deployment
    - Swagger Documentation
    """,
    version="1.0.0",
    contact={
        "name": "Gaurav Tiwari",
        "url": "https://github.com/gauravtiwari0259"
    }
)


with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

w = model["weights"]
b = model["bias"]
mu = model["mu"]
sd = model["sd"]



def sigmoid(z):
    return 1 / (1 + np.exp(-z))



class Patient(BaseModel):

    Pregnancies: float = Field(
        description="Number of pregnancies"
    )

    Glucose: float = Field(
        description="Plasma glucose concentration"
    )

    BloodPressure: float = Field(
        description="Diastolic blood pressure (mm Hg)"
    )

    SkinThickness: float = Field(
        description="Triceps skin fold thickness"
    )

    Insulin: float = Field(
        description="2-Hour serum insulin"
    )

    BMI: float = Field(
        description="Body Mass Index"
    )

    Age: float = Field(
        description="Age in years"
    )

    DiabetesPedigreeFunction: float = Field(
        description="Diabetes pedigree function"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "Pregnancies": 2,
                "Glucose": 120,
                "BloodPressure": 70,
                "SkinThickness": 20,
                "Insulin": 80,
                "BMI": 25.5,
                "Age": 30,
                "DiabetesPedigreeFunction": 0.5
            }
        }
    }


@app.get("/", tags=["General"])
def home():
    return {
        "message": "Diabetes Detection API Running",
        "model": "Custom Logistic Regression",
        "version": "1.0.0"
    }


@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Predict Diabetes",
    description="""
    Predicts whether a patient is diabetic using
    a custom Logistic Regression model trained on
    the Pima Indians Diabetes Dataset.
    """
)
def predict(patient: Patient):

    X = np.array([
        patient.Pregnancies,
        patient.Glucose,
        patient.BloodPressure,
        patient.SkinThickness,
        patient.Insulin,
        patient.BMI,
        patient.Age,
        patient.DiabetesPedigreeFunction
    ])


    X = (X - mu) / sd


    prob = sigmoid(np.dot(X, w) + b)

    pred = 1 if prob >= 0.5 else 0

    return {
        "probability": round(float(prob), 4),
        "prediction": pred,
        "result": "Diabetic" if pred else "Non-Diabetic",
        "model": "Custom Logistic Regression",
        "version": "1.0.0"
    }