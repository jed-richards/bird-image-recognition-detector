import pytest
import requests

def test_index_route():
    response = requests.get("http://localhost:5000/")
    assert response.json() == {"result": 1}
