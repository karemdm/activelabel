"""Ultralytics YOLO worker (detection, instance and pose, classification): ``python yolo.py spec.json``.

Standalone, like the other workers: it runs in the training environment (which has no ActiveLabel) and talks
to the runner with the same events (``log``, ``warning``, ``setup``, ``progress``, ``resumed``, ``epoch``,
``done``, ``error``) and the same files in ``output_dir`` (``best.pt``, ``last.pt``, ``results.json``). It
needs the ``ultralytics`` package, whose licence (AGPL-3.0) the person accepts before it is installed
(``training/licences.py``).

The spec's images become an Ultralytics dataset in a temporary folder (images linked or copied, one label
file per image, ``data.yaml``; for classification one folder per class in each part), the model is trained by
Ultralytics' own trainer, and callbacks turn its epochs into ``epoch`` events:

- **Resize policy**: ``letterbox`` is what Ultralytics does itself; ``stretch`` and ``center_crop`` photos are
  fitted to the square input first (:func:`prepare_dataset`), labels included, like the other workers.
- **Augmentation**: the resolved techniques become Ultralytics' hyperparameters (:func:`augmentation_args`);
  mosaic is the model's own ``mosaic`` and ``close_mosaic`` options. A technique Ultralytics cannot apply
  (blur, grayscale) is reported in a ``warning``.
- **Selection**: the best epoch is the one with the best selection metric, not Ultralytics' own fitness, so
  ``best.pt`` is a copy of ``last.pt`` taken at that epoch; patience and early stopping follow the same
  metric.
- **Classification**: the scores are those of the classification worker (accuracy, macro F1, per class and the
  confusion matrix ``[true][predicted]``), computed from Ultralytics' confusion matrix.
- **Resume**: ``last.pt`` is Ultralytics' checkpoint (optimizer, EMA, schedule); ``yolo_state.json`` keeps the
  best epoch and the run's ``digest``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps

# Ultralytics keeps its settings and the weights it downloads in a user folder: use the run's temporary one
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(os.environ.get("TMPDIR") or Path.cwd()) / "yolo-config"))

try:
    from ultralytics import YOLO, settings
except ImportError:  # the licence was not accepted, or the package is missing: reported in train()
    YOLO = None
    settings = None

try:
    from activelabel.training.workers import classification as scoring
    from activelabel.training.workers import options
except ImportError:  # run as a script in the training environment (no ActiveLabel): the sibling files
    import classification as scoring  # type: ignore[no-redef]
    import options  # type: ignore[no-redef]

METRICS = ("map50_95", "map50", "accuracy", "macro_f1")
TASKS = ("detection", "instance", "keypoints", "classification")
#: Selection metrics of each task.
TASK_METRICS = {"classification": ("accuracy", "macro_f1")}
STATE_FILE = "yolo_state.json"
RUN_NAME = "yolo"
#: Ultralytics optimizer names of the ``optimizer`` option.
OPTIMIZERS = {"sgd": "SGD", "adam": "Adam", "adamw": "AdamW"}
#: Backbone layers kept frozen when only the head trains (``start`` = ``transfer``).
FROZEN_LAYERS = 10
FINAL_LR_FRACTION = 0.01  # Ultralytics' default ``lrf``
UNSUPPORTED_TECHNIQUES = ("blur", "grayscale")
Scores = dict[str, Any]


class SpecError(ValueError):
    """The spec cannot be used (the message is readable)."""


# ---------------------------------------------------------------------------
# Pure helpers (no ultralytics)


def emit(event: str, **fields: Any) -> None:
    """Print one event line.

    Args:
        event: Event name (``log``, ``warning``, ``setup``, ``progress``, ``resumed``, ``epoch``, ``done`` or
            ``error``).
        **fields: Extra fields of the JSON object.
    """
    print(json.dumps({"event": event, **fields}), flush=True)


def load_spec(path: str | Path) -> dict[str, Any]:
    """Read and check ``spec.json``.

    Args:
        path: Path of the spec file.

    Returns:
        The spec as a dict.

    Raises:
        SpecError: missing keys, an unsupported task, family, start or metric, or no images.
    """
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("task", "model", "family", "start", "params", "classes", "parts", "output_dir", "metric"):
        if key not in spec:
            raise SpecError(f"spec.json has no '{key}'.")
    if spec["task"] not in TASKS:
        raise SpecError(f"This worker trains {', '.join(TASKS)}, not {spec['task']}.")
    if spec.get("fold") == "all":
        raise SpecError("Training all folds comes later: choose one fold.")
    if spec["family"] != "ultralytics":
        raise SpecError(f"This worker trains Ultralytics models, not {spec['family']}.")
    if spec["start"] not in ("transfer", "finetune", "scratch"):
        raise SpecError(f"Unknown start '{spec['start']}'.")
    allowed = TASK_METRICS.get(spec["task"], ("map50_95", "map50"))
    if spec["metric"] not in allowed:
        raise SpecError(f"The selection metric must be one of: {', '.join(allowed)}.")
    parts = spec["parts"]
    if not parts.get("train") or not parts.get("val"):
        raise SpecError("Training needs images in the train and validation parts.")
    if spec["task"] == "keypoints" and int(spec.get("num_keypoints") or 0) < 1:
        raise SpecError("The labels have no keypoints.")
    return dict(spec)


def device_arg(device: str) -> str:
    """The Ultralytics ``device`` of a torch device name.

    Args:
        device: ``cpu``, ``mps``, ``cuda`` or ``cuda:N`` (see ``options.pick_device``).

    Returns:
        ``cpu``, ``mps`` or the GPU index (``"0"``).
    """
    if device.startswith("cuda"):
        return device.partition(":")[2] or "0"
    return device


def weights_name(model_id: str, start: str) -> str:
    """The file Ultralytics loads for a model: pretrained weights, or the architecture for ``scratch``.

    Args:
        model_id: catalogue id (``yolov8n``, ``yolo11s-seg``...).
        start: ``transfer``, ``finetune`` or ``scratch``.

    Returns:
        ``<id>.pt`` (downloaded by Ultralytics) or ``<id>.yaml`` (random weights).
    """
    return f"{model_id}.{'yaml' if start == 'scratch' else 'pt'}"


def schedule_args(scheduler: str) -> tuple[dict[str, Any], str | None]:
    """The Ultralytics learning rate options of a ``scheduler`` choice.

    Args:
        scheduler: ``none``, ``cosine``, ``step``, ``onecycle`` or ``plateau``.

    Returns:
        ``(options, unsupported)``: ``cos_lr`` and ``lrf``, and the scheduler's name when Ultralytics cannot
        do it (it then falls back to its linear decay), else ``None``.
    """
    if scheduler == "none":
        return {"cos_lr": False, "lrf": 1.0}, None
    if scheduler == "cosine":
        return {"cos_lr": True, "lrf": FINAL_LR_FRACTION}, None
    return {"cos_lr": False, "lrf": FINAL_LR_FRACTION}, scheduler


def augmentation_args(
    techniques: list[dict[str, Any]], mosaic: bool, task: str, num_keypoints: int
) -> tuple[dict[str, Any], list[int] | None, list[str]]:
    """Ultralytics' augmentation hyperparameters for the resolved techniques (everything else is off).

    Args:
        techniques: the resolved entries ``{"name", "params"}`` of the spec.
        mosaic: the model's ``mosaic`` option (four images in one).
        task: ``detection``, ``instance`` or ``keypoints``.
        num_keypoints: points per object (pose).

    Returns:
        ``(args, flip_idx, unsupported)``: the ``train`` arguments, the left/right point permutation for the
        horizontal flip of a pose (``None`` for other tasks) and the names of techniques Ultralytics cannot
        apply.
    """
    args: dict[str, Any] = {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "mosaic": 1.0 if mosaic else 0.0,
    }
    flip_idx = list(range(num_keypoints)) if task == "keypoints" else None
    unsupported: list[str] = []
    for entry in techniques:
        name, params = entry["name"], entry.get("params") or {}
        if name == "hflip":
            args["fliplr"] = float(params.get("p", 0.5))
            if flip_idx is not None:
                for a, b in params.get("flip_pairs") or []:
                    flip_idx[a], flip_idx[b] = b, a
        elif name == "vflip":
            args["flipud"] = float(params.get("p", 0.5))
        elif name == "rotation":
            args["degrees"] = float(params.get("degrees", 0.0))
        elif name == "scale_jitter":
            low, high = float(params.get("scale_min", 1.0)), float(params.get("scale_max", 1.0))
            args["scale"] = max(1.0 - low, high - 1.0)  # Ultralytics zooms by a gain in [1 - s, 1 + s]
        elif name == "translate":
            args["translate"] = float(params.get("fraction", 0.0))
        elif name == "color_jitter":
            args["hsv_h"] = float(params.get("hue", 0.0))
            args["hsv_s"] = float(params.get("saturation", 0.0))
            args["hsv_v"] = float(params.get("brightness", 0.0))
        elif name in UNSUPPORTED_TECHNIQUES:
            unsupported.append(name)
    return args, flip_idx, unsupported


def object_extent(obj: dict[str, Any]) -> list[float] | None:
    """Relative ``[x1, y1, x2, y2]`` of an object: its box, else the extent of its polygons or visible points.

    Args:
        obj: spec object with a relative ``bbox`` ``[x, y, w, h]``, ``polygons`` or ``keypoints``.

    Returns:
        The box, or ``None`` when the object has no shape.
    """
    if obj.get("bbox"):
        x, y, w, h = obj["bbox"][:4]
        return [x, y, x + w, y + h]
    if obj.get("polygons"):
        points = [p for ring in obj["polygons"] for p in ring]
    else:
        points = [p for p in obj.get("keypoints") or [] if (p[2] if len(p) > 2 else 2) > 0]
    if not points:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _clip(value: float) -> float:
    """A coordinate limited to the photo.

    Args:
        value: relative coordinate.

    Returns:
        The value between 0 and 1.
    """
    return min(max(value, 0.0), 1.0)


def move_point(x: float, y: float, layout: tuple[int, int, int, int], size: int) -> tuple[float, float]:
    """A relative point of the photo put in the fitted square input.

    Args:
        x: relative abscissa in the photo.
        y: relative ordinate in the photo.
        layout: ``(scaled width, scaled height, left, top)`` of ``options.fit_layout``.
        size: side of the square input.

    Returns:
        The point relative to the input (outside 0..1 when the fit cuts it away).
    """
    width, height, left, top = layout
    return (x * width + left) / size, (y * height + top) / size


def move_object(obj: dict[str, Any], layout: tuple[int, int, int, int] | None, size: int) -> dict[str, Any]:
    """An object with its coordinates put in the fitted square input (relative to the input).

    Args:
        obj: spec object (relative to the photo).
        layout: ``(scaled width, scaled height, left, top)`` of ``options.fit_layout`` for the photo;
            ``None`` keeps the coordinates (the photo is not fitted).
        size: side of the square input.

    Returns:
        A copy whose ``bbox``, ``polygons`` and ``keypoints`` are relative to the input; a point the fit
        puts outside the input is clipped (a keypoint becomes not labelled).
    """
    if layout is None:
        return obj
    moved = dict(obj)
    if obj.get("bbox"):
        x, y, w, h = obj["bbox"][:4]
        (x1, y1), (x2, y2) = move_point(x, y, layout, size), move_point(x + w, y + h, layout, size)
        moved["bbox"] = [_clip(x1), _clip(y1), _clip(x2) - _clip(x1), _clip(y2) - _clip(y1)]
    if obj.get("polygons"):
        moved["polygons"] = [
            [[_clip(a) for a in move_point(p[0], p[1], layout, size)] for p in ring]
            for ring in obj["polygons"]
        ]
    if obj.get("keypoints"):
        points = []
        for p in obj["keypoints"]:
            x, y = move_point(p[0], p[1], layout, size)
            inside = 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
            points.append([x, y, p[2] if len(p) > 2 else 2] if inside else [0.0, 0.0, 0])
        moved["keypoints"] = points
    return moved


def label_lines(objects: list[dict[str, Any]], task: str, num_keypoints: int) -> list[str]:
    """The lines of an Ultralytics label file for the objects of an image (coordinates relative to it).

    Args:
        objects: spec objects (``label`` class index plus ``bbox``, ``polygons`` or ``keypoints``).
        task: ``detection``, ``instance`` or ``keypoints``.
        num_keypoints: points per object (pose).

    Returns:
        ``class cx cy w h`` for detection; ``class x1 y1 x2 y2 ...`` per polygon (at least 3 points) for
        instance segmentation; ``class cx cy w h x y visibility ...`` for pose. Objects without the shape
        or whose box has no area are left out.
    """
    lines = []
    for obj in objects:
        box = object_extent(obj)
        if box is None:
            continue
        x1, y1, x2, y2 = (_clip(v) for v in box)
        if x2 <= x1 or y2 <= y1:
            continue
        cls = int(obj["label"])
        if task == "instance":
            for ring in obj.get("polygons") or []:
                if len(ring) >= 3:
                    coords = " ".join(f"{_clip(p[0]):.6f} {_clip(p[1]):.6f}" for p in ring)
                    lines.append(f"{cls} {coords}")
            continue
        head = f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
        if task == "keypoints":
            points = list(obj.get("keypoints") or [])
            if len(points) != num_keypoints:
                continue
            head += "".join(f" {p[0]:.6f} {p[1]:.6f} {int(p[2] if len(p) > 2 else 2)}" for p in points)
        lines.append(head)
    return lines


def data_yaml(
    root: Path, classes: list[str], parts: list[str], num_keypoints: int, flip_idx: list[int] | None
) -> str:
    """The text of ``data.yaml`` (JSON-quoted scalars are valid YAML).

    Args:
        root: dataset folder.
        classes: class names, in index order.
        parts: the parts that have images (``train``, ``val``, maybe ``test``).
        num_keypoints: points per object (pose); 0 for the other tasks.
        flip_idx: left/right permutation of the points (pose), else ``None``.

    Returns:
        The YAML text.
    """
    lines = [f"path: {json.dumps(str(root))}"]
    lines += [f"{name}: images/{name}" for name in parts]
    lines.append("names:")
    lines += [f"  {i}: {json.dumps(name)}" for i, name in enumerate(classes)]
    if num_keypoints:
        lines.append(f"kpt_shape: [{num_keypoints}, 3]")
        lines.append(f"flip_idx: {json.dumps(flip_idx or list(range(num_keypoints)))}")
    return "\n".join(lines) + "\n"


def per_class_scores(metric: Any, classes: list[str], support: Any = None) -> dict[str, dict[str, float]]:
    """Per-class AP of an Ultralytics ``Metric`` as ``{name: {"ap50", "ap50_95", "support"}}``.

    Args:
        metric: an Ultralytics ``Metric`` (``ap_class_index``, ``ap50`` and ``ap``).
        classes: class names, in index order.
        support: instances per class (``nt_per_class``), or ``None``.

    Returns:
        The classes that had ground truth.
    """
    out: dict[str, dict[str, float]] = {}
    ap50, ap = list(metric.ap50), list(metric.ap)
    for k, index in enumerate(list(metric.ap_class_index)):
        index = int(index)
        if index >= len(classes):
            continue
        count = int(support[index]) if support is not None and index < len(support) else 0
        out[classes[index]] = {"ap50": float(ap50[k]), "ap50_95": float(ap[k]), "support": count}
    return out


def classification_scores(matrix: Any, classes: list[str]) -> Scores:
    """The scores of the classification worker from Ultralytics' confusion matrix.

    Args:
        matrix: Ultralytics' ``matrix[predicted][true]`` counts (any array of ``len(classes)`` rows).
        classes: class names, in index order.

    Returns:
        ``accuracy``, ``macro_f1``, ``per_class``, ``confusion_matrix`` (``[true][predicted]``), ``images``.
    """
    n = len(classes)
    rows = [[int(round(float(matrix[p][t]))) for p in range(n)] for t in range(n)]
    return {
        "accuracy": scoring.accuracy(rows),
        "macro_f1": scoring.macro_f1(rows),
        "per_class": scoring.per_class(rows, classes),
        "confusion_matrix": rows,
        "images": sum(sum(r) for r in rows),
    }


def scores_of(metrics: Any, classes: list[str], task: str, matrix: Any = None) -> Scores:
    """The scores of an Ultralytics metrics object, in the shape the other workers report.

    Args:
        metrics: ``DetMetrics``, ``SegmentMetrics``, ``PoseMetrics`` (``box``, ``seg`` or ``pose``) or
            ``ClassifyMetrics``.
        classes: class names, in index order.
        task: ``detection``, ``instance``, ``keypoints`` or ``classification``.
        matrix: the validator's ``ConfusionMatrix`` (classification; its ``matrix`` is read).

    Returns:
        ``map50``, ``map50_95`` and ``per_class`` of the boxes, plus ``mask_map50``, ``mask_map50_95``
        (instance) or ``kp_map50``, ``kp_map50_95`` (keypoints); for classification those of
        :func:`classification_scores`.

    Raises:
        ValueError: a classification without a confusion matrix.
    """
    if task == "classification":
        found = matrix if matrix is not None else getattr(metrics, "confusion_matrix", None)
        if found is None:
            raise ValueError("Ultralytics gave no confusion matrix for this classification.")
        return classification_scores(getattr(found, "matrix", found), classes)
    box = metrics.box
    scores: Scores = {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": per_class_scores(box, classes, getattr(metrics, "nt_per_class", None)),
    }
    extra = {"instance": ("mask", "seg"), "keypoints": ("kp", "pose")}.get(task)
    if extra is not None:
        part = getattr(metrics, extra[1])
        scores[f"{extra[0]}_map50"] = float(part.map50)
        scores[f"{extra[0]}_map50_95"] = float(part.map)
    return scores


def epoch_scores(scores: Scores) -> dict[str, float]:
    """The ``val_*`` fields of an ``epoch`` event.

    Args:
        scores: the dict of :func:`scores_of`.

    Returns:
        The scores without ``per_class``, prefixed with ``val_``.
    """
    return {f"val_{k}": v for k, v in scores.items() if k != "per_class"}


def report_step(epoch: int, epochs: int, step: int, steps: int) -> None:
    """Tell the runner about the progress of a training epoch (about twenty times per epoch).

    Args:
        epoch: the epoch (from 1).
        epochs: how many epochs the run has.
        step: the batch just done (from 1).
        steps: batches in the epoch.
    """
    if step == steps or step % max(1, steps // 20) == 0:
        emit("progress", phase="training", epoch=epoch, epochs=epochs, step=step, steps=steps)


# ---------------------------------------------------------------------------
# The dataset


def link_or_copy(source: Path, target: Path) -> None:
    """Put a file in the dataset: a hard link when the disk allows it, else a copy (never a symbolic link).

    Args:
        source: the photo.
        target: its place in the dataset.
    """
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare_dataset(spec: dict[str, Any], root: Path, flip_idx: list[int] | None) -> Path:
    """Write the Ultralytics dataset of a spec under ``root`` and return its ``data.yaml``.

    Photos are linked (or copied) unless the resize policy is ``stretch`` or ``center_crop``: those are
    fitted to the square input here (``options.fit_image``) and their labels move with them. ``letterbox``
    is left to Ultralytics.

    Args:
        spec: the job spec (``parts``, ``classes``, ``task``, ``num_keypoints``, ``params``).
        root: an empty folder for the dataset.
        flip_idx: left/right permutation of the points (pose).

    Returns:
        The path of ``data.yaml``.
    """
    task, classes = spec["task"], spec["classes"]
    points = int(spec.get("num_keypoints") or 0)
    size, policy = int(spec["params"]["image_size"]), str(spec["params"]["resize"])
    names = [name for name in ("train", "val", "test") if spec["parts"].get(name)]
    if task == "classification":
        return prepare_classification(spec, root, names, size, policy)
    for name in names:
        (root / "images" / name).mkdir(parents=True)
        (root / "labels" / name).mkdir(parents=True)
        for k, item in enumerate(spec["parts"][name]):
            source = Path(item["path"])
            stem = f"{k:06d}"
            objects = item.get("objects") or []
            if policy == "letterbox":
                link_or_copy(source, root / "images" / name / f"{stem}{source.suffix.lower() or '.jpg'}")
            else:
                with Image.open(source) as opened:
                    photo = ImageOps.exif_transpose(opened).convert("RGB")
                    layout = options.fit_layout(photo.width, photo.height, size, policy)
                    options.fit_image(photo, size, policy).save(
                        root / "images" / name / f"{stem}.jpg", quality=95
                    )
                objects = [move_object(o, layout, size) for o in objects]
            text = "\n".join(label_lines(objects, task, points))
            (root / "labels" / name / f"{stem}.txt").write_text(
                text + ("\n" if text else ""), encoding="utf-8"
            )
    yaml = root / "data.yaml"
    yaml.write_text(data_yaml(root, classes, names, points, flip_idx), encoding="utf-8")
    return yaml


def prepare_classification(
    spec: dict[str, Any], root: Path, names: list[str], size: int, policy: str
) -> Path:
    """Write an Ultralytics classification dataset: ``<part>/<class index>/<image>`` under ``root``.

    The class folders are named by the zero-padded index (``0000``...), so Ultralytics, which sorts them by
    name, numbers the classes as the project does; every part has every class folder.

    Args:
        spec: the job spec (``parts`` with ``label`` class indices, ``classes``).
        root: an empty folder for the dataset.
        names: the parts that have images (``train``, ``val``, ``test``).
        size: side of the square input.
        policy: the resize policy (``letterbox`` is left to Ultralytics).

    Returns:
        ``root``, what Ultralytics takes as ``data`` for classification.
    """
    for name in names:
        for index in range(len(spec["classes"])):
            (root / name / f"{index:04d}").mkdir(parents=True)
        for k, item in enumerate(spec["parts"][name]):
            source, folder = Path(item["path"]), root / name / f"{int(item['label']):04d}"
            if policy == "letterbox":
                link_or_copy(source, folder / f"{k:06d}{source.suffix.lower() or '.jpg'}")
            else:
                with Image.open(source) as opened:
                    photo = ImageOps.exif_transpose(opened).convert("RGB")
                    options.fit_image(photo, size, policy).save(folder / f"{k:06d}.jpg", quality=95)
    return root


# ---------------------------------------------------------------------------
# Training


def train_args(
    spec: dict[str, Any], device: str, use_amp: bool, data: Path, out: Path, augmentation: dict[str, Any]
) -> dict[str, Any]:
    """The keyword arguments of ``YOLO.train`` for a spec.

    Args:
        spec: the job spec.
        device: torch device name (``cpu``, ``mps``, ``cuda`` or ``cuda:N``).
        use_amp: whether mixed precision is on.
        data: path of ``data.yaml``.
        out: the run's output folder (Ultralytics works in ``out/yolo``).
        augmentation: the arguments of :func:`augmentation_args`.

    Returns:
        The arguments. Patience is left off here: the worker stops by the selection metric itself.
    """
    p = spec["params"]
    scheduler, _ = schedule_args(str(p["scheduler"]))
    args = {
        "data": str(data),
        "epochs": int(p["epochs"]),
        "imgsz": int(p["image_size"]),
        "batch": int(p["batch_size"]),
        "device": device_arg(device),
        "workers": 4 if device.startswith("cuda") else 0,
        "optimizer": OPTIMIZERS[str(p["optimizer"])],
        "lr0": float(p["lr"]),
        "weight_decay": float(p["weight_decay"]),
        "warmup_epochs": float(p["warmup_epochs"]),
        "close_mosaic": int(p["close_mosaic"]),
        "patience": 0,
        "seed": int(spec.get("seed", p["seed"])),
        "amp": use_amp,
        "pretrained": spec["start"] != "scratch",
        "project": str(out),
        "name": RUN_NAME,
        "exist_ok": True,
        "plots": False,
        "verbose": False,
        "val": True,
        "save": True,
        **scheduler,
        **augmentation,
    }
    if spec["start"] == "transfer":
        args["freeze"] = FROZEN_LAYERS
    return args


def read_state(out: Path) -> dict[str, Any]:
    """The state a resumed training needs, written every epoch.

    Args:
        out: the run's output folder.

    Returns:
        ``epoch``, ``digest`` and ``best`` (``{"epoch", "scores"}`` or empty); empty without a state file.
    """
    path = out / STATE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class EpochRecorder:
    """Turns Ultralytics' epochs into ``epoch`` events, ``last.pt``, ``best.pt`` and the resume state.

    One epoch is finished when Ultralytics has both validated it and saved its checkpoint; the two callbacks
    can come in either order, so the epoch is closed by whichever comes second.

    Attributes:
        best: ``{"epoch", "scores"}`` of the best epoch so far (empty before the first).
        ran: epochs finished, counting those of the interrupted training when resumed.
    """

    def __init__(self, spec: dict[str, Any], out: Path, state: dict[str, Any] | None = None) -> None:
        """Start recording; a resumed training gives the state it left.

        Args:
            spec: the job spec.
            out: the run's output folder.
            state: the content of ``yolo_state.json`` of the interrupted training, if resuming.
        """
        self.spec = spec
        self.out = out
        self.metric = str(spec["metric"])
        self.patience = int(spec["params"]["patience"])
        self.best: dict[str, Any] = dict((state or {}).get("best") or {})
        self.ran = int((state or {}).get("epoch") or 0)
        self.steps_done = 0
        self._started = time.monotonic()
        self._scores: Scores | None = None
        self._row: dict[str, Any] = {}
        self._saved = False

    def on_train_epoch_start(self, trainer: Any) -> None:
        """Ultralytics callback: a new epoch begins.

        Args:
            trainer: the Ultralytics trainer.
        """
        self.steps_done = 0
        self._started = time.monotonic()
        self._scores, self._saved = None, False

    def on_train_batch_end(self, trainer: Any) -> None:
        """Ultralytics callback: a training batch is done.

        Args:
            trainer: the Ultralytics trainer.
        """
        self.steps_done += 1
        report_step(trainer.epoch + 1, trainer.epochs, self.steps_done, len(trainer.train_loader))

    def on_fit_epoch_end(self, trainer: Any) -> None:
        """Ultralytics callback: the epoch was trained and validated.

        Args:
            trainer: the Ultralytics trainer.
        """
        validator = trainer.validator
        self._scores = scores_of(
            validator.metrics,
            self.spec["classes"],
            self.spec["task"],
            getattr(validator, "confusion_matrix", None),
        )
        losses = trainer.label_loss_items(trainer.tloss)
        rates = list(getattr(trainer, "lr", {}).values())
        self._row = {
            "train_loss": float(sum(float(v) for v in losses.values())),
            "lr": float(rates[0]) if rates else None,
        }
        self._close(trainer)

    def on_model_save(self, trainer: Any) -> None:
        """Ultralytics callback: ``last.pt`` of the epoch was written.

        Args:
            trainer: the Ultralytics trainer.
        """
        self._saved = True
        self._close(trainer)

    def _close(self, trainer: Any) -> None:
        """Finish the epoch once it is validated and saved: checkpoints, state and the ``epoch`` event.

        Args:
            trainer: the Ultralytics trainer (``last`` is its checkpoint, ``stop`` ends the training).
        """
        if self._scores is None or not self._saved:
            return
        scores, epoch = self._scores, trainer.epoch + 1
        self._scores, self._saved = None, False
        improved = not self.best or scores[self.metric] > self.best["scores"][self.metric]
        shutil.copy2(trainer.last, self.out / "last.pt")
        if improved:
            self.best = {"epoch": epoch, "scores": scores}
            shutil.copy2(trainer.last, self.out / "best.pt")
        self.ran = epoch
        (self.out / STATE_FILE).write_text(
            json.dumps({"epoch": epoch, "digest": str(self.spec.get("digest") or ""), "best": self.best}),
            encoding="utf-8",
        )
        emit(
            "epoch",
            epoch=epoch,
            checkpoint=True,
            epochs=int(self.spec["params"]["epochs"]),
            train_loss=self._row.get("train_loss"),
            **epoch_scores(scores),
            lr=self._row.get("lr"),
            seconds=round(time.monotonic() - self._started, 2),
            best=improved,
        )
        if self.patience and epoch - self.best["epoch"] >= self.patience:
            emit("log", message=f"Early stopping: no improvement for {self.patience} epoch(s).")
            trainer.stop = True


def warn_unsupported(spec: dict[str, Any], techniques: list[str]) -> None:
    """Tell the runner about options Ultralytics cannot follow exactly.

    Args:
        spec: the job spec (``params`` and ``start``).
        techniques: names of the augmentation techniques Ultralytics cannot apply.
    """
    p = spec["params"]
    for name in techniques:
        emit(
            "warning",
            message=f"Ultralytics cannot apply the '{name}' augmentation: it is left out.",
            code="yolo_technique_unsupported",
            params={"technique": name},
        )
    _, unsupported = schedule_args(str(p["scheduler"]))
    if unsupported:
        emit(
            "warning",
            message=f"Ultralytics has no '{unsupported}' schedule: its linear decay is used instead.",
            code="yolo_scheduler_unsupported",
            params={"scheduler": unsupported},
        )
    if spec["start"] == "finetune" and float(p.get("backbone_lr_factor", 1.0)) != 1.0:  # old configs only
        emit(
            "warning",
            message="Ultralytics trains the whole network at one learning rate: the backbone factor "
            "is ignored.",
            code="yolo_backbone_lr_ignored",
            params={},
        )


def resume_checkpoint(out: Path, digest: str, data: Path) -> Path:
    """Put the interrupted training's checkpoint where Ultralytics resumes from, pointing at the new dataset.

    Args:
        out: the run's output folder (``last.pt`` and ``yolo_state.json``).
        digest: the ``digest`` of the spec (what defines the training).
        data: the ``data.yaml`` rebuilt for this process (the old temporary dataset is gone).

    Returns:
        The path of the checkpoint to resume (``out/yolo/weights/last.pt``).

    Raises:
        options.CodedFailure: no checkpoint (``resume_no_checkpoint``), or the training changed since it
            stopped (``resume_config_changed``).
    """
    last = out / "last.pt"
    state = read_state(out)
    if not last.exists() or not state:
        raise options.CodedFailure("This run has no checkpoint to resume from.", "resume_no_checkpoint")
    if state.get("digest") != digest:
        raise options.CodedFailure(
            "The configuration or the data of this run changed since it stopped: start a new run.",
            "resume_config_changed",
        )
    # weights_only=False: last.pt is written by this worker from Ultralytics' own checkpoint
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    checkpoint["train_args"]["data"] = str(data)
    checkpoint["train_args"]["project"] = str(out)
    target = out / RUN_NAME / "weights" / "last.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)
    return target


def train(spec: dict[str, Any]) -> dict[str, Any]:
    """Train with Ultralytics, keep the best epoch by the selection metric, evaluate the test part.

    Writes ``best.pt``, ``last.pt``, ``yolo_state.json`` and ``results.json`` to ``output_dir``.

    Args:
        spec: The checked job spec.

    Returns:
        The results: ``metric``, ``best_epoch``, ``epochs_run``, ``stopped_early``, ``val`` and ``test``
        (``None`` without a test part).

    Raises:
        SpecError: ultralytics is not installed.
        options.DeviceError: The device the run asked for is not available.
        options.CodedFailure: Nothing to resume from, or the training changed since it stopped.
        options.OutOfMemory: The GPU ran out of memory.
    """
    if YOLO is None:
        raise SpecError("The ultralytics package is not installed in the training environment.")
    p, classes, task = spec["params"], spec["classes"], spec["task"]
    out = Path(spec["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if settings is not None:
        settings.update({"sync": False})  # no telemetry
    device = options.pick_device(torch, str(spec.get("device") or "auto"))
    emit("log", message=f"Device: {device}.")
    use_amp, amp_note = options.amp_enabled(str(p["amp"]), device)
    if amp_note:
        emit("warning", message=amp_note, code="amp_needs_cuda", params={})
    emit("setup", device=device, amp=use_amp, batch_size=int(p["batch_size"]))
    techniques = spec["augmentations"]
    points = int(spec.get("num_keypoints") or 0)
    augmentation, flip_idx, unsupported = augmentation_args(techniques, bool(p["mosaic"]), task, points)
    warn_unsupported(spec, unsupported)
    emit("log", message=f"Resize: {p['resize']}. Mosaic: {'on' if p['mosaic'] else 'off'}.")
    root = Path(tempfile.mkdtemp(prefix="yolo-data-", dir=Path.cwd()))
    try:
        data = prepare_dataset(spec, root, flip_idx)
        state = read_state(out) if spec.get("resume") else None
        recorder = EpochRecorder(spec, out, state)
        args = train_args(spec, device, use_amp, data, out, augmentation)
        if spec.get("resume"):
            model = YOLO(str(resume_checkpoint(out, str(spec.get("digest") or ""), data)))
            args = {"resume": True}
            emit("resumed", epoch=recorder.ran)
            emit("log", message=f"Resumed after epoch {recorder.ran} of {int(p['epochs'])}.")
        else:
            model = YOLO(weights_name(spec["model"], spec["start"]))
        for name in ("on_train_epoch_start", "on_train_batch_end", "on_fit_epoch_end", "on_model_save"):
            model.add_callback(name, getattr(recorder, name))
        try:
            model.train(**args)
        except Exception as exc:
            if options.is_out_of_memory(exc):
                raise options.OutOfMemory(
                    "The GPU ran out of memory: lower the batch size or the image size, or train on the CPU."
                ) from None
            raise
        if not recorder.best:
            raise RuntimeError("Ultralytics finished without completing an epoch.")
        epochs = int(p["epochs"])
        results: dict[str, Any] = {
            "metric": spec["metric"],
            "best_epoch": recorder.best["epoch"],
            "epochs_run": recorder.ran,
            "stopped_early": recorder.ran < epochs,
            "val": {**recorder.best["scores"], "classes": classes},
            "test": None,
        }
        if spec["parts"].get("test"):
            emit("progress", phase="evaluating", epoch=recorder.ran, epochs=epochs)
            metrics = YOLO(str(out / "best.pt")).val(
                data=str(data),
                split="test",
                imgsz=int(p["image_size"]),
                batch=int(p["batch_size"]),
                device=device_arg(device),
                plots=False,
                verbose=False,
            )
            results["test"] = {**scores_of(metrics, classes, task), "classes": classes}
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(
            out / RUN_NAME, ignore_errors=True
        )  # Ultralytics' own copies: last.pt and best.pt are kept
    (out / "results.json").write_text(json.dumps(results), encoding="utf-8")
    return results


def main(argv: list[str]) -> int:
    """Entry point.

    Args:
        argv: Command line; ``argv[1]`` is the path of the spec JSON file.

    Returns:
        Process exit code: 0 on success, 1 after an ``error`` event.
    """
    try:
        spec = load_spec(argv[1])
        results = train(spec)
    except Exception as exc:  # noqa: BLE001 - reported to the runner as an event
        emit("error", message=str(exc) or type(exc).__name__, **options.coded(exc))
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1
    emit("done", results=results)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main(sys.argv))
