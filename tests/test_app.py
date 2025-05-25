import json
from pathlib import Path
import pytest
from unittest.mock import patch

from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.testing = True
    return app.test_client()

def test_separate_success(client, tmp_path):
    dummy_audio_path = tmp_path / "test.wav"
    dummy_audio_path.write_bytes(b"fake wav data")

    output_dir = tmp_path / "separated" / "test"
    output_dir.mkdir(parents=True)

    fake_output = [output_dir / "vocals.wav", output_dir / "accompaniment.wav"]
    for path in fake_output:
        path.write_bytes(b"fake separated data")

    with patch("app.check_audio_file"), \
         patch("app.ffmpeg_convert", return_value=dummy_audio_path), \
         patch("app.separator.separate", return_value=fake_output):

        response = client.post(
            "/separate",
            json={"path": str(dummy_audio_path)}
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "separated_paths" in data
        assert sorted(data["separated_paths"]) == sorted([str(p) for p in fake_output])

def test_separate_missing_path(client):
    response = client.post("/separate", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "400 Bad Request: JSON payload must contain 'path'"

def test_separate_invalid_json(client):
    response = client.post("/separate", data="not json", content_type="application/json")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

def test_separate_model_exception(client, tmp_path):
    dummy_audio_path = tmp_path / "test.wav"
    dummy_audio_path.write_bytes(b"fake data")

    with patch("app.check_audio_file"), \
         patch("app.ffmpeg_convert", return_value=dummy_audio_path), \
         patch("app.separator.separate", side_effect=RuntimeError("oops")):

        response = client.post("/separate", json={"path": str(dummy_audio_path)})
        assert response.status_code == 500
        data = response.get_json()
        assert "Model inference failed" in data["error"]
