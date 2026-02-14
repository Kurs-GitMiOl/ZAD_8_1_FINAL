# Imports
from pydantic import BaseModel, Field
from fastapi import APIRouter
import joblib
import numpy as np
from pathlib import Path

# Load the pre-trained model safely regardless of current working directory
# Loads the trained model safely, no matter the current folder

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "iris_model.joblib"

# Contrlol info print
# print("Loading model from:", MODEL_PATH)
model = joblib.load(MODEL_PATH)

# Create router
router = APIRouter()

# Input data schema
class IrisInput(BaseModel):
    sepal_length: float = Field(..., ge=4.3, le=7.9, description="Sepal length must be between 4.3 and 7.9 cm")
    sepal_width: float = Field(..., ge=2.0, le=4.4, description="Sepal width must be between 2.0 and 4.4 cm")
    petal_length: float = Field(..., ge=1.0, le=6.9, description="Petal length must be between 1.0 and 6.9 cm")
    petal_width: float = Field(..., ge=0.1, le=2.5, description="Petal with must be between 0.1 and 2.5 cm")



@router.post("/predict")
def predict(data: IrisInput):
    """
    Prediction endpoint for iris flower classification.
    It takes flower features in JSON format, predicts the flower class,
    and returns both the class number and the class name.
    """

    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(features)[0]

    target_names = ["setosa", "versicolor", "virginica"]

    return {
        "prediction_class": int(prediction),
        "prediction_name": target_names[prediction]
    }

@router.post("/predict_is_setosa")
def predict_is_setosa(data: IrisInput):
    """
    Prediction endpoint that checks if the iris flower is Setosa.
    It takes flower features in JSON format and returns True if the predicted class is Setosa,
    otherwise False.
    """
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(features)[0]  # 0=setosa, 1=versicolor, 2=virginica

    return {
        "is_setosa": bool(prediction == 0)
    }


@router.post("/describe_input")
def describe_input(data: IrisInput):
    """
    Endpoint that describes iris flower features.
    It takes flower features in JSON, calculates the minimum, maximum, and mean values, and returns them.
    """

    features = [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    return {
        "min": min(features),
        "max": max(features),
        "mean": round(sum(features)/len(features), 3)
    }


@router.post("/predict_proba")
def predict_proba(input: IrisInput):
    """
    Return prediction probabilities for each Iris flower class.
    The endpoint takes flower features and returns a probability for each class.
    """

    features = [[
        input.sepal_length,
        input.sepal_width,
        input.petal_length,
        input.petal_width
    ]]

    probabilities = model.predict_proba(features)[0]
    return {
        "setosa": round(float(probabilities[0]), 3),
        "versicolor": round(float(probabilities[1]), 3),
        "virginica": round(float(probabilities[2]), 3)
    }


@router.get("/describe_input_get")
def describe_input_get(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    """
    Describe Iris input values using query parameters.
    The endpoint returns min, max, and mean values of the input features.
    """

    features = [sepal_length, sepal_width, petal_length, petal_width]
    return {
        "min": min(features),
        "max": max(features),
        "mean": round(sum(features)/len(features), 3)
    }

@router.get("/model_info")
def model_info():
    """
    Endpoint that returns basic information about the used ML model.
    It shows model type, kernel type, and if probability is enabled.
    """
    return {
        "model_type": "SVC",
        "kernel": "linear",
        "probability": True
    }

@router.get("/status")
def status_check():
    """
    Simple status check endpoint.
    Returns 'ok' if the API is running.
    """
    return {"status": "ok"}
