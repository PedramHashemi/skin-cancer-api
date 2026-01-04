"""Tests."""

import os
os.environ.setdefault("TESTING", "true")

from fastapi.testclient import TestClient
from app.main import app
from io import BytesIO
from PIL import Image
import pytest

client = TestClient(app)

@pytest.fixture
def create_dummy_image(file_type="jpeg"):
    """creating a dummy image file."""
    img_byte_arr = BytesIO()
    img = Image.new('RGB', (100, 100))
    img.save(img_byte_arr, format=file_type)
    img_byte_arr.seek(0)
    return [(
        "skin_image",
        ("test_image.jpg", img_byte_arr.getvalue(), "image/jpeg"),
    )]

@pytest.fixture
def create_dummy_text():
    """creating a dummy text file."""
    text_byte_arr = BytesIO(b"This is a test text file.")
    text_byte_arr.name = "test_file.txt"
    text_byte_arr.seek(0)
    return [(
        "skin_image",
        ("test_file.txt", text_byte_arr.getvalue(), "text/plain"),
    )]

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

def test_image_upload_success(create_dummy_image):
    response = client.post("/image/upload/", files=create_dummy_image)
    print(response.json())
    assert response.status_code == 200
    assert response.json()['class'] in range(7)

def test_image_upload_fail(create_dummy_text):
    response = client.post("/image/upload/", files=create_dummy_text)
    assert response.status_code == 400
    assert response.json().get("detail") == "Invalid file type. Only jpg, jpeg"