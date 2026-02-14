from fastapi.testclient import TestClient

from app.main import app
import math

# Create a test client for the FastAPI app
client = TestClient(app)

def test_predict():
    """
    Basic test for the /predict endpoint:
    Sends a sample Iris flower JSON and checks if the response
    returns status 200 and contains the expected prediction fields
    """
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.2,
            "petal_length": 2.4,
            "petal_width": 1.2
        }
    )

    # Check HTTP status code
    assert response.status_code == 200

    data = response.json()

    # Check response structure
    assert "prediction_class" in data
    assert "prediction_name" in data

# ---------------------------------------------------------------

def test_predictis_is_setosa():
    """
    Test /predict_is_setosa endpoint.
    Checks if API returns a boolean value indicating whether the flower is setosa.
    """
    response = client.post("/predict_is_setosa", json={
        "sepal_length": 5.0,
        "sepal_width": 3.4,
        "petal_length": 1.5,
        "petal_width": 0.2
    })

    # Check HTTP status code
    assert response.status_code == 200

    data = response.json()

    # Check response structure
    assert "is_setosa" in data

    # Check value type
    assert isinstance(data["is_setosa"], bool)



# ---------------------------------------------------------------

def test_predict_missing_field():
    """
    Test /predict endpoint with missing field.
    Checks if API returns 422  when a required input is missing.
    """
    response = client.post("/predict", json={
        "sepal_length": 5.0,
        "sepal_width": 3.4,
        "petal_length": 1.5
    })

    # Check HTTP status code
    assert response.status_code == 422

#------------------------------------------------------------------

def test_predict_edge_cases():
    """
    Test /predict with minimum and maximum feature values
    from Iris dataset
    """
    min_features = {
        "sepal_length": 4.3,
        "sepal_width": 2.0,
        "petal_length": 1.0,
        "petal_width": 0.1
    }

    response_min = client.post("/predict", json=min_features)
    assert response_min.status_code == 200
    data_min = response_min.json()
    assert "prediction_class" in data_min
    assert "prediction_name" in data_min

    # Test /predict with maximum feature values from Iris dataset
    max_features = {
        "sepal_length": 7.9,
        "sepal_width": 4.4,
        "petal_length": 6.9,
        "petal_width": 2.5
    }

    response_max = client.post("/predict", json=max_features)
    assert response_max.status_code == 200
    data_max = response_max.json()
    assert "prediction_class" in data_max
    assert "prediction_name" in data_max

#----------------------------------------------------------------

def test_predict_proba():
    """
    Test /predict_proba:
    Send flower data and check probabilities sum to ~1
    """
    response = client.post(
        "/predict_proba",
        json={
            "sepal_length": 6.0,
            "sepal_width": 3.9,
            "petal_length": 4.5,
            "petal_width": 2.5
        }
    )

    assert response.status_code == 200

    data = response.json()

    # Check probabilities exist
    assert "setosa" in data
    assert "versicolor" in data
    assert "virginica" in data

    # Check probabilities sum to ~1.0
    total = data["setosa"] + data["versicolor"] + data["virginica"]
    assert abs(total - 1.0) < 0.01

# -------------------------------------------------------------------

def test_describe_input():
    """
    Test /describe_input
    Send flower data and check min, max, mean values
    """
    response = client.post(
        "/describe_input",
        json={
            "sepal_length": 5.0,
            "sepal_width": 2.8,
            "petal_length": 4.5,
            "petal_width": 1.2
        }
    )

    # Check HTTP status code
    assert response.status_code == 200

    data = response.json()

    # Check response structure
    assert "min" in data
    assert "max" in data
    assert "mean" in data

    # Expected values
    values = [5.0, 2.8, 4.5, 1.2]
    expected_min = min(values)
    expected_max = max(values)
    expected_mean = sum(values) / len(values)

    # Check values are correct
    assert math.isclose(data["min"], expected_min, rel_tol=1e-2)
    assert math.isclose(data["max"], expected_max, rel_tol=1e-2)
    assert math.isclose(data["mean"], expected_mean, rel_tol=1e-2)

#------------------------------------------------------------------------

def test_describe_input_get():
    """
    Test /describe_input_get
    send flower data as query params and check min, max, mean
    """
    response = client.get(
        "/describe_input_get",
        params={
            "sepal_length": 5.5,
            "sepal_width": 2.8,
            "petal_length": 4.0,
            "petal_width": 1.5
        }
    )

    # Check HTTP status code
    assert response.status_code == 200

    data = response.json()

    # Check response structure
    assert "min" in data
    assert "max" in data
    assert "mean" in data

    # Check values are correct
    values = [5.5, 2.8, 4.0, 1.5]
    expected_min = min(values)
    expected_max = max(values)
    expected_mean = sum(values) / len(values)

    assert math.isclose(data["min"], expected_min, rel_tol=1e-2)
    assert math.isclose(data["max"], expected_max, rel_tol=1e-2)
    assert math.isclose(data["mean"], expected_mean, rel_tol=1e-2)


#-------------------------------------------------------------------------

def test_describe_input_invalid():
    """
    Test /describe_input with invalid data
    Check it returns 422 error
    """
    response = client.post("/describe_input", json={
        "sepal_length": 2.0,
        "sepal_width": "error",
        "petal_length": 3.0,
        "petal_width": 1.0
    })

    assert response.status_code == 422

# -----------------------------------------------------------------------

def test_model_info():
    """
    Check if endpoint returns basic model configuration
    """
    response = client.get("/model_info")

    # Check HTTP status code
    assert response.status_code == 200

    data = response.json()

    # Check response structure
    assert "model_type" in data
    assert "kernel" in data
    assert "probability" in data

    # Check returned values
    assert data["model_type"] == "SVC"
    assert data["kernel"] == "linear"
    assert data["probability"] is True

# ----------------------------------------------------------------------

def test_status():
    """
    Test /status endpoint.
    Checks if API returns status ok.
    """
    response = client.get("/status")

    # Check HTTP status code
    assert response.status_code == 200

    data = response.json()

    # Check response
    assert data["status"] == "ok"