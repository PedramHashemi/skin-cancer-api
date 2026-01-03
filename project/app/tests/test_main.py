from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Skin Cancer Detection Tool API",
        "environment": "dev"
    }

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_invalid_endpoint():
    response = client.get("/invalid-endpoint")
    assert response.status_code == 404

def test_method_not_allowed():
    response = client.post("/health")
    assert response.status_code == 405

def test_upload_no_file():
    response = client.post("/image/upload/")
    assert response.status_code == 422

# def test_upload_invalid_file():
    