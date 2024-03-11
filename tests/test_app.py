import pytest
import requests

PREDICT_ROUTE = "http://localhost:5000/predict/"


def test_predict_data_flow():
    response = requests.post(
        url=PREDICT_ROUTE,
        files={"image": open("data/test/ABBOTTS BABBLER/1.jpg", "rb")},
    )
    prediction = [
        "species",
        "predicted_id",
        "accuracy",
        "user_image",
        "predicted_image",
    ]
    data = dict(response.json())
    for k in prediction:
        assert k in data
