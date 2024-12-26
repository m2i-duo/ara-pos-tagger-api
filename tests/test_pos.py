from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_pos_tag():
    response = client.post("/pos-tag", json={"text": "your test text"})
    assert response.status_code == 200
    assert "tagged_text" in response.json()