import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "engine" in data
    assert "version" in data


def test_sentiment_single_endpoint(client):
    payload = {"text": "Khana mast tha par delivery late thi"}
    response = client.post("/api/v1/sentiment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment_label"] in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    assert "food" in data["aspects"]
    assert "delivery" in data["aspects"]
    assert data["linguistics"]["is_code_mixed"] is True
    assert data["latency_ms"] >= 0.0


def test_sentiment_batch_endpoint(client):
    payload = {
        "texts": [
            "Phone badiya hai",
            "Bilkul bakwas experience",
            "Wah bhai 3 ghante me deliver kiya shabaash"
        ]
    }
    response = client.post("/api/v1/sentiment/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 3
    assert len(data["predictions"]) == 3


def test_normalization_endpoint(client):
    payload = {"text": "ye bht axha h"}
    response = client.post("/api/v1/normalize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["normalized_text"] == "yeh bohot accha hai"
    assert data["changes_count"] > 0


def test_validation_error_empty_text(client):
    response = client.post("/api/v1/sentiment", json={"text": ""})
    assert response.status_code == 422
