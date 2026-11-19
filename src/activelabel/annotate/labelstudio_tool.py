"""Label Studio adapter of the annotation tools, over plain HTTP (``requests``).

Why not FiftyOne's ``labelstudio`` backend: it needs ``label-studio-sdk`` (not installed) and
passes the API key as a legacy token, but Label Studio 1.23 disables legacy tokens by default and
hands out a "Personal Access Token", which is a JWT *refresh* token. So this module talks to the
REST API itself:

* auth: the token is exchanged at ``POST /api/token/refresh/`` for a short-lived access token
  (``Authorization: Bearer``); if that is refused the token is tried as a legacy token
  (``Authorization: Token``). A new exchange is made for every operation and again after a 401.
* send: one project per session with a labeling config by task (Choices, RectangleLabels,
  PolygonLabels, KeyPointLabels), the images uploaded as files named ``<image_id>.<ext>``, and the
  current ground truth created as annotations of the tasks (so they are edited, not re-drawn).
* pull: ``GET /api/projects/<id>/export?exportType=JSON``; the latest non-cancelled annotation of
  each task is converted to internal labels (percent values -> 0..1).
* close: optionally deletes the project.

Keypoints: Label Studio has no "pose object", so each point is a ``KeyPointLabels`` label named
``"<class> <n>"``; points of the same class are grouped into objects in order of appearance.
The token never appears in logs, files or exception messages.
"""

from __future__ import annotations

import math
import mimetypes
import re
import uuid
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

import requests

from activelabel.annotate.base import GROUND_TRUTH, Sent, ToolError, ToolSettings
from activelabel.labels import Label
from activelabel.projects import Project
from activelabel.tasks import Task

TIMEOUT = 30
UPLOAD_TIMEOUT = 300
BATCH = 20  # files per upload request
UPLOAD_PREFIX = re.compile(r"^[0-9a-fA-F]{8}-")
FROM, TO = "label", "image"


def _clamp(v: float) -> float:
    return min(1.0, max(0.0, v))


def labeling_config(task: Task, classes: list[str], keypoints: int = 1) -> str:
    """Label Studio labeling config (XML) for a task and the project classes."""
    img = f'<Image name="{TO}" value="$image" zoom="true"/>'
    if task is Task.CLASSIFICATION:
        body = "".join(f"<Choice value={quoteattr(c)}/>" for c in classes)
        inner = f'<Choices name="{FROM}" toName="{TO}" choice="single" showInline="true">{body}</Choices>'
    else:
        tag = {
            Task.DETECTION: "RectangleLabels",
            Task.SEGMENTATION: "PolygonLabels",
            Task.KEYPOINTS: "KeyPointLabels",
        }[task]
        names = (
            [f"{c} {i + 1}" for c in classes for i in range(keypoints)] if task is Task.KEYPOINTS else classes
        )
        body = "".join(f"<Label value={quoteattr(n)}/>" for n in names)
        extra = ' strokeWidth="2"' if task is Task.KEYPOINTS else ""
        inner = f'<{tag} name="{FROM}" toName="{TO}"{extra}>{body}</{tag}>'
    return f"<View>{img}{inner}</View>"


def _rid() -> str:
    return uuid.uuid4().hex[:10]


def _result(kind: str, value: dict[str, Any], w: int, h: int) -> dict[str, Any]:
    return {
        "id": _rid(),
        "type": kind,
        "from_name": FROM,
        "to_name": TO,
        "original_width": w,
        "original_height": h,
        "image_rotation": 0,
        "origin": "manual",
        "value": value,
    }


def to_results(labels: list[Label], task: Task, size: tuple[int, int]) -> list[dict[str, Any]]:
    """Internal labels -> Label Studio annotation results (percent coordinates)."""
    w, h = size
    out: list[dict[str, Any]] = []
    for lb in labels:
        if task is Task.CLASSIFICATION:
            if not out:
                out.append(_result("choices", {"choices": [lb.label]}, w, h))
        elif task is Task.DETECTION and lb.bbox:
            x, y, bw, bh = lb.bbox
            value = {
                "x": x * 100,
                "y": y * 100,
                "width": bw * 100,
                "height": bh * 100,
                "rotation": 0,
                "rectanglelabels": [lb.label],
            }
            out.append(_result("rectanglelabels", value, w, h))
        elif task is Task.SEGMENTATION and lb.polygons:
            for ring in lb.polygons:
                pts = [[p[0] * 100, p[1] * 100] for p in ring]
                value = {"points": pts, "closed": True, "polygonlabels": [lb.label]}
                out.append(_result("polygonlabels", value, w, h))
        elif task is Task.KEYPOINTS and lb.keypoints:
            for i, kp in enumerate(lb.keypoints):
                if len(kp) > 2 and kp[2] <= 0:
                    continue  # not labeled: leave it for the annotator
                value = {
                    "x": kp[0] * 100,
                    "y": kp[1] * 100,
                    "width": 0.6,
                    "keypointlabels": [f"{lb.label} {i + 1}"],
                }
                out.append(_result("keypointlabels", value, w, h))
    return out


def _rect_box(v: dict[str, Any]) -> list[float]:
    """Percent rectangle -> relative ``[x, y, w, h]`` (rotated ones become their enclosing box)."""
    x, y, w, h = (float(v[k]) / 100 for k in ("x", "y", "width", "height"))
    angle = math.radians(float(v.get("rotation") or 0))
    if angle:
        c, s = math.cos(angle), math.sin(angle)
        pts = [(x + dx * c - dy * s, y + dx * s + dy * c) for dx, dy in ((0, 0), (w, 0), (w, h), (0, h))]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    x0, y0 = _clamp(x), _clamp(y)
    return [x0, y0, _clamp(x + w) - x0, _clamp(y + h) - y0]


def from_results(results: list[dict[str, Any]], task: Task, keypoints: int = 0) -> list[Label]:
    """Label Studio annotation results -> internal labels (relative coordinates)."""
    labels: list[Label] = []
    objects: dict[str, list[dict[int, list[float]]]] = {}  # class -> objects {point index: [x, y, vis]}
    order: list[str] = []
    for res in results:
        kind, val = res.get("type"), res.get("value") or {}
        if task is Task.CLASSIFICATION and kind == "choices" and val.get("choices"):
            return [Label(str(val["choices"][0]))]
        if task is Task.DETECTION and kind == "rectanglelabels" and val.get("rectanglelabels"):
            labels.append(Label(str(val["rectanglelabels"][0]), bbox=_rect_box(val)))
        elif task is Task.SEGMENTATION and kind == "polygonlabels" and val.get("polygonlabels"):
            ring = [[_clamp(float(x) / 100), _clamp(float(y) / 100)] for x, y in val.get("points") or []]
            if len(ring) >= 3:
                xs, ys = [p[0] for p in ring], [p[1] for p in ring]
                box = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                labels.append(Label(str(val["polygonlabels"][0]), bbox=box, polygons=[ring]))
        elif task is Task.KEYPOINTS and kind == "keypointlabels" and val.get("keypointlabels"):
            cls, _, num = str(val["keypointlabels"][0]).rpartition(" ")
            if not num.isdigit() or not cls:
                continue
            idx = int(num) - 1
            point = [_clamp(float(val["x"]) / 100), _clamp(float(val["y"]) / 100), 2.0]
            groups = objects.setdefault(cls, [])
            if cls not in order:
                order.append(cls)
            target = next((g for g in groups if idx not in g), None)
            if target is None:
                target = {}
                groups.append(target)
            target[idx] = point
    for cls in order:
        for group in objects[cls]:
            n = max(keypoints, max(group) + 1)
            pts = [group.get(i, [0.0, 0.0, 0.0]) for i in range(n)]
            seen = [p for p in pts if p[2]]
            xs, ys = [p[0] for p in seen], [p[1] for p in seen]
            box = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            labels.append(Label(cls, bbox=box, keypoints=pts))
    return labels


def _image_id(task: dict[str, Any]) -> str:
    """Image id of a task, from the uploaded file name (``<hash>-<image_id>.<ext>``)."""
    data = task.get("data") or {}
    value = str(data.get("image") or "") if isinstance(data, dict) else ""
    name = value.split("?", 1)[0].rsplit("/", 1)[-1]
    return Path(UPLOAD_PREFIX.sub("", name)).stem


class _Client:
    """Minimal authenticated Label Studio REST client."""

    def __init__(self, settings: ToolSettings) -> None:
        url = settings.url.strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ToolError(
                "Label Studio address must start with http:// or https:// (e.g. http://localhost:8080)."
            )
        if not settings.secret:
            raise ToolError("Label Studio needs an API token (Account & Settings > Personal Access Token).")
        self.url = url
        self._token = settings.secret
        self._http = requests.Session()
        self._authenticate()

    def _authenticate(self) -> None:
        self._http.headers.pop("Authorization", None)
        try:
            r = self._http.post(
                f"{self.url}/api/token/refresh/", json={"refresh": self._token}, timeout=TIMEOUT
            )
        except requests.RequestException as err:
            raise self._unreachable(err) from None
        access = r.json().get("access") if r.status_code == 200 else None
        if access:
            self._http.headers["Authorization"] = f"Bearer {access}"
        else:  # not a JWT refresh token: a legacy token
            self._http.headers["Authorization"] = f"Token {self._token}"

    def _unreachable(self, err: Exception) -> ToolError:
        return ToolError(
            f"Cannot reach Label Studio at {self.url}: check the address and that it is running."
        )

    def request(self, method: str, path: str, *, timeout: int = TIMEOUT, **kw: Any) -> requests.Response:
        """Call the API; re-authenticate once on 401; map failures to :class:`ToolError`."""
        for attempt in (0, 1):
            try:
                r = self._http.request(method, f"{self.url}{path}", timeout=timeout, **kw)
            except requests.RequestException as err:
                raise self._unreachable(err) from None
            if r.status_code == 401 and attempt == 0:
                self._authenticate()
                continue
            break
        if r.status_code in (401, 403):
            raise ToolError(
                "Label Studio refused the token: create a new Personal Access Token and save it again."
            )
        if r.status_code == 404:
            raise ToolError(
                f"Label Studio answered 'not found' for {path.split('?')[0]}: wrong address or deleted."
            )
        if r.status_code >= 400:
            raise ToolError(f"Label Studio error {r.status_code} on {path.split('?')[0]}.")
        return r


def _labels_from_fo(value: Any, task: Task) -> list[Label]:
    from activelabel.annotate.fo_labels import labels_from_fo

    return labels_from_fo(value, task)


def _image_size(path: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return int(im.width), int(im.height)


class LabelStudioTool:
    """Label Studio over its REST API."""

    name = "labelstudio"

    def check(self, settings: ToolSettings) -> str:
        """Call whoami; return ``Connected to <url> as <email>``."""
        client = _Client(settings)
        try:
            r = client.request("GET", "/api/current-user/whoami")
        except ToolError as err:
            if "not found" not in str(err):
                raise
            r = client.request("GET", "/api/users/whoami")
        who = r.json().get("email") or r.json().get("username") or "unknown user"
        return f"Connected to {client.url} as {who}"

    def send(
        self, dataset: Any, image_ids: list[str], project: Project, settings: ToolSettings, session_id: str
    ) -> Sent:
        """Create a project, upload the images and pre-load the ground truth as annotations."""
        client = _Client(settings)
        view = dataset.select_by("image_id", image_ids)
        ids = list(view.values("image_id"))
        paths = list(view.values("filepath"))
        gts = list(view.values(GROUND_TRUTH))
        sizes = {i: _image_size(p) for i, p in zip(ids, paths, strict=True)}
        labels = {
            i: _labels_from_fo(v, project.task) if v is not None else []
            for i, v in zip(ids, gts, strict=True)
        }
        points = max([len(lb.keypoints) for ls in labels.values() for lb in ls if lb.keypoints] or [1])
        title = f"ActiveLabel {project.id} - {project.name} [{session_id}]"
        config = labeling_config(project.task, list(project.classes), points)
        pid = int(
            client.request("POST", "/api/projects", json={"title": title, "label_config": config}).json()[
                "id"
            ]
        )
        try:
            self._upload(client, pid, ids, paths)
            tasks = client.request("GET", f"/api/tasks?project={pid}&page_size=10000&fields=all").json()
            tasks = tasks["tasks"] if isinstance(tasks, dict) else tasks
            for t in tasks:
                image_id = _image_id(t)
                results = to_results(labels.get(image_id, []), project.task, sizes.get(image_id, (0, 0)))
                if results:
                    client.request("POST", f"/api/tasks/{t['id']}/annotations", json={"result": results})
        except Exception as err:
            try:
                client.request("DELETE", f"/api/projects/{pid}")
            except Exception:  # noqa: S110 - best effort, the original error matters
                pass
            if isinstance(err, ToolError):
                raise
            raise ToolError(f"Label Studio error while sending: {type(err).__name__}") from None
        links = [{"label": "Label Studio project", "url": f"{client.url}/projects/{pid}/data"}]
        return Sent(
            links=links, ref={"project_id": pid, "keypoints": points if project.task is Task.KEYPOINTS else 0}
        )

    @staticmethod
    def _upload(client: _Client, pid: int, ids: list[str], paths: list[str]) -> None:
        for start in range(0, len(ids), BATCH):
            handles = []
            try:
                files = []
                for image_id, path in zip(
                    ids[start : start + BATCH], paths[start : start + BATCH], strict=True
                ):
                    fh = open(path, "rb")  # noqa: SIM115 - closed in the finally below
                    handles.append(fh)
                    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
                    files.append(("file", (f"{image_id}{Path(path).suffix.lower()}", fh, mime)))
                client.request(
                    "POST",
                    f"/api/projects/{pid}/import?commit_to_project=true",
                    files=files,
                    timeout=UPLOAD_TIMEOUT,
                )
            finally:
                for fh in handles:
                    fh.close()

    def pull(
        self,
        dataset: Any,
        image_ids: list[str],
        project: Project,
        settings: ToolSettings,
        ref: dict[str, Any],
    ) -> dict[str, list[Label]]:
        """Export the project as JSON and convert the latest annotation of every task."""
        pid = ref.get("project_id")
        if not pid:
            raise ToolError("This session has no Label Studio project.")
        client = _Client(settings)
        r = client.request(
            "GET",
            f"/api/projects/{pid}/export?exportType=JSON&download_all_tasks=true",
            timeout=UPLOAD_TIMEOUT,
        )
        out: dict[str, list[Label]] = {i: [] for i in image_ids}
        for task in r.json():
            image_id = _image_id(task)
            if image_id not in out:
                continue
            done = [a for a in task.get("annotations") or [] if not a.get("was_cancelled")]
            if not done:
                continue
            latest = max(done, key=lambda a: str(a.get("updated_at") or a.get("created_at") or ""))
            out[image_id] = from_results(
                latest.get("result") or [], project.task, int(ref.get("keypoints") or 0)
            )
        return out

    def close(self, dataset: Any, settings: ToolSettings, ref: dict[str, Any], delete_remote: bool) -> None:
        """Delete the Label Studio project when asked; a project that is already gone is fine."""
        pid = ref.get("project_id")
        if not (delete_remote and pid):
            return
        try:
            _Client(settings).request("DELETE", f"/api/projects/{pid}")
        except ToolError as err:
            if "not found" not in str(err):
                raise
