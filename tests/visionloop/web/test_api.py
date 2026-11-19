import time

from api_client import make_client, upload
from sample_images import make_corrupt, make_image


def wait_job(client, project_id, job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        job = next(j for j in client.get(f"/api/projects/{project_id}/jobs").json() if j["id"] == job_id)
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_project_lifecycle(tmp_path):
    client, _ = make_client(tmp_path)
    created = client.post("/api/projects", json={"name": "Cats", "task": "detection", "classes": ["cat"]})
    assert created.status_code == 200
    project = created.json()
    assert project["id"] == 1 and project["steps"] == {
        "images": "empty",
        "annotations": "none",
        "curation": "not_analyzed",
    }
    assert client.post("/api/projects", json={"name": "cats", "task": "detection"}).status_code == 400
    assert client.post("/api/projects", json={"name": "X", "task": "nope"}).status_code == 422
    changed = client.patch("/api/projects/1", json={"description": "d", "task": "segmentation"}).json()
    assert changed["task"] == "segmentation" and changed["description"] == "d"
    assert client.get("/api/projects/99").status_code == 404
    wrong = client.request("DELETE", "/api/projects/1", json={"confirm_name": "Dogs"})
    assert wrong.status_code == 400
    assert client.request("DELETE", "/api/projects/1", json={"confirm_name": "Cats"}).json() == {"deleted": 1}
    assert client.get("/api/projects").json() == []


def test_images_upload_list_exclude_delete(tmp_path):
    client, _ = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "classification"})
    paths = [make_image(tmp_path / f"in/{i}.png", seed=i) for i in range(3)]
    paths.append(make_corrupt(tmp_path / "in/bad.jpg"))
    report = upload(client, 1, paths, ["a/0.png", "a/1.png", "2.png", "bad.jpg"]).json()
    assert report["added"] == 3
    assert [r["reason"] for r in report["rejected"]] == ["corrupt"]
    assert upload(client, 1, paths[:1]).json()["duplicates"] == ["0.png"]
    page = client.get("/api/projects/1/images", params={"limit": 2}).json()
    assert page["total"] == 3 and len(page["items"]) == 2 and "file" not in page["items"][0]
    first = page["items"][0]["id"]
    assert client.get(f"/api/projects/1/images/{first}/thumb").headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/projects/1/images/{first}/file").status_code == 200
    assert client.get("/api/projects/1/images/nope/thumb").status_code == 404
    assert client.post(
        "/api/projects/1/images/exclude", json={"ids": [first], "reason": "blurry"}
    ).json() == {"changed": 1}
    assert client.get("/api/projects/1/images", params={"status": "excluded"}).json()["total"] == 1
    assert client.get("/api/projects/1").json()["counts"]["excluded"] == 1
    assert client.post("/api/projects/1/images/restore", json={"ids": [first]}).json() == {"changed": 1}
    assert client.post("/api/projects/1/images/delete", json={"ids": [first]}).status_code == 400
    assert client.post("/api/projects/1/images/delete", json={"ids": [first], "confirm": True}).json() == {
        "deleted": 1
    }
    assert client.get("/api/projects/1").json()["counts"]["images"] == 2


def test_folder_import_is_a_job(tmp_path):
    root = tmp_path / "data"
    make_image(root / "set/a.png")
    client, _ = make_client(tmp_path, import_root=root)
    client.post("/api/projects", json={"name": "P", "task": "detection"})
    job = client.post("/api/projects/1/images/folder", json={"path": str(root / "set")}).json()
    done = wait_job(client, 1, job["id"])
    assert done["status"] == "done" and done["result"]["added"] == 1
    outside = client.post("/api/projects/1/images/folder", json={"path": str(tmp_path)}).json()
    assert wait_job(client, 1, outside["id"])["status"] == "failed"


def test_folder_import_off_without_root(tmp_path):
    client, _ = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "detection"})
    response = client.post("/api/projects/1/images/folder", json={"path": str(tmp_path)})
    assert response.status_code == 400 and "VISIONLOOP_IMPORT_ROOT" in response.json()["detail"]


def test_cancel_unknown_job(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.post("/api/jobs/nope/cancel").status_code == 404
