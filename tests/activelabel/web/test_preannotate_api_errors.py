from api_client import make_client

from activelabel.storage import write_json
from activelabel.training.models import ModelStore


def test_a_yolo_model_cannot_pre_annotate_and_the_message_has_a_code(tmp_path):
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "detection", "classes": ["a"]})
    folder = ModelStore(state.store.dir(1)).path(1)
    folder.mkdir(parents=True)
    write_json(folder / "model.json", {"number": 1, "model": "yolov8n", "task_kind": "detection"})
    response = client.post("/api/projects/1/preannotations", json={"model": 1, "task": 1, "threshold": 0.5})
    body = response.json()
    assert response.status_code == 400 and body["code"] == "preannotation_kind_unsupported"
    assert body["params"] == {"kind": "detection"}
    assert "later" not in body["detail"] and "YOLO" in body["detail"]
