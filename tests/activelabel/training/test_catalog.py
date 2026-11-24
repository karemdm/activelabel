from activelabel.training.catalog import MODELS, PROJECT_TYPES, TASK_KINDS, get_model, models_for


def test_catalog_consistency():
    ids = [m.id for m in MODELS]
    assert len(ids) == len(set(ids))
    for m in MODELS:
        assert m.packages and m.task_kind in TASK_KINDS and m.types
        for spec in m.param_specs():
            assert spec.check(spec.default) == spec.default, (m.id, spec.name)
            if spec.kind == "choice":
                assert spec.default in spec.choices
        assert m.param("image_size").default == m.image_size
    for t in PROJECT_TYPES:
        assert models_for(t)
    assert {m.task_kind for m in models_for("segmentation")} == {"semantic", "instance"}
    for wanted in (
        "resnet18",
        "vit_b_16",
        "ssdlite320_mobilenet_v3_large",
        "yolov8n",
        "yolo11l-seg",
        "yolo11m-pose",
    ):
        assert get_model(wanted)
    assert "ultralytics" in get_model("yolov8s").packages
