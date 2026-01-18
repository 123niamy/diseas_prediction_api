from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("disease_model.joblib")

# Define the input data model
class PatientData(BaseModel):
    age: int
    gender: int
    blood_pressure: int
    cholesterol: int
    glucose: int
    bmi: float
    family_history: int
    smoking: int

app = FastAPI()

@app.post("/predict")
def predict_disease(data: PatientData):
    # Prepare input for prediction
    features = np.array([[data.age, data.gender, data.blood_pressure, data.cholesterol,
                          data.glucose, data.bmi, data.family_history, data.smoking]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0, 1]
    return {"disease_prediction": int(prediction), "probability": float(probability)}
