"""The catalogue declares, per family and task, the hyperparameters its worker really reads.

The workers run in the separate training environment, so they are read here as source (``ast``), not imported:
every key a worker takes from ``params`` must be declared by each model it trains, and every parameter a model
declares must be read by its worker (``start`` is read from the run spec itself).
"""

import ast
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from activelabel import i18n
from activelabel.errors import CodedError
from activelabel.training import catalog
from activelabel.training.catalog import (
    COMMON_PARAMS,
    MODELS,
    PROJECT_TYPES,
    TASK_KINDS,
    ParamSpec,
    get_model,
    models_for,
)
from activelabel.training.config import ConfigError, RunConfig, validate
from activelabel.training.runner import (
    FAMILY_WORKER_FILES,
    RESOLVED_PARAMS,
    WORKER_FILES,
    WORKERS_DIR,
    worker_key,
)

#: Families listed in the catalogue that no worker trains yet (the API refuses them, the form hides them).
NOT_TRAINABLE: set[str] = set()


def _is_params(node: ast.expr) -> bool:
    """``p`` or ``spec["params"]``: where a worker keeps the hyperparameters."""
    if isinstance(node, ast.Name):
        return node.id == "p"
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "params"
    )


#: Helpers of another worker that a worker calls with its spec (the regression worker builds and optimises its
#: network with the classification worker's functions, #619).
BORROWED = {"regression.py": ("classification.py", ("make_optimizer",))}


def _reads(script: Path) -> tuple[set[str], set[str]]:
    """Keys a worker reads from its hyperparameters, and keys it reads from the run spec itself."""
    params: set[str] = set()
    spec: set[str] = set()
    tree = ast.parse(script.read_text(encoding="utf-8"))
    roots: list[ast.AST] = [tree]
    if script.name in BORROWED:
        other, names = BORROWED[script.name]
        borrowed = ast.parse((script.parent / other).read_text(encoding="utf-8"))
        roots += [f for f in borrowed.body if isinstance(f, ast.FunctionDef) and f.name in names]
    for node in (n for root in roots for n in ast.walk(root)):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            key = node.slice.value
            if isinstance(key, str) and _is_params(node.value):
                params.add(key)
            elif isinstance(key, str) and isinstance(node.value, ast.Name) and node.value.id == "spec":
                spec.add(key)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _is_params(node.func.value)
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            params.add(node.args[0].value)
    return params, spec


TRAINABLE = [m for m in MODELS if m.family not in NOT_TRAINABLE]
#: Options a worker still reads from older configurations although its models no longer declare them (#666).
LEGACY_READS = {"ultralytics": {"backbone_lr_factor"}}


@pytest.mark.parametrize("model", TRAINABLE, ids=lambda m: m.id)
def test_every_declared_parameter_is_read_by_the_worker(model):
    params, spec = _reads(
        WORKERS_DIR / (FAMILY_WORKER_FILES.get(model.family) or WORKER_FILES[model.task_kind])
    )
    declared = {p.name for p in model.param_specs()}
    assert "start" in spec  # the start is a field of the run, not of params
    assert "augmentations" in spec  # the augmentation level reaches the worker as a list of techniques
    unread = declared - params - {"start"} - set(RESOLVED_PARAMS)
    assert not unread, f"{model.id}: declared but not read: {unread}"
    legacy = LEGACY_READS.get(model.family, set())  # read only to warn about an older configuration
    assert params - legacy <= declared, f"{model.id}: read but not declared: {params - declared - legacy}"


def test_yolo_worker_reads_the_options_only_yolo_models_declare():
    read = _reads(WORKERS_DIR / FAMILY_WORKER_FILES["ultralytics"])[0]
    assert {"mosaic", "close_mosaic"} <= read
    assert all(worker_key(m) == "ultralytics" for m in MODELS if m.family == "ultralytics")
    torchvision_read = set().union(*(_reads(WORKERS_DIR / name)[0] for name in set(WORKER_FILES.values())))
    assert not {"mosaic", "close_mosaic"} & torchvision_read  # the other workers know no mosaic option


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.id)
def test_defaults_are_valid_for_each_model(model):
    for spec in model.param_specs():
        assert spec.check(spec.default) == spec.default, f"{model.id}.{spec.name}"
        if spec.min is not None:
            assert spec.min <= spec.max


@pytest.mark.parametrize(
    ("model_id", "size", "low", "high"),
    [
        ("resnet18", 224, 32, 2048),
        ("efficientnet_b0", 224, 32, 2048),
        ("efficientnet_b3", 288, 32, 2048),
        ("vit_b_16", 224, 224, 224),  # fixed position embeddings
        ("fasterrcnn_resnet50_fpn", 640, 32, 2048),
        ("ssdlite320_mobilenet_v3_large", 320, 320, 320),  # fixed input
        ("deeplabv3_resnet50", 512, 32, 2048),
        ("yolov8n", 640, 32, 2048),
        ("yolo11s-seg", 640, 32, 2048),
    ],
)
def test_image_size_per_family(model_id, size, low, high):
    spec = get_model(model_id).param("image_size")
    assert (spec.default, spec.min, spec.max) == (size, low, high)


def test_family_defaults():
    yolo = {p.name: p.default for p in get_model("yolov8n").param_specs()}
    assert (yolo["optimizer"], yolo["lr"], yolo["weight_decay"]) == ("sgd", 0.01, 0.0005)
    assert {p.name for p in get_model("yolov8n").params} == {"mosaic", "close_mosaic"}
    detection = {p.name: p.default for p in get_model("retinanet_resnet50_fpn").param_specs()}
    assert (detection["optimizer"], detection["lr"], detection["batch_size"]) == ("sgd", 0.005, 4)
    assert get_model("vit_b_16").param("lr").default == 0.0001
    assert get_model("resnet18").param("lr").default == 0.001
    semantic = {p.name for p in get_model("deeplabv3_resnet50").params}
    # no score threshold nor backbone layers (the worker reads neither): only the imbalance options
    assert semantic == {"class_weights", "loss"}


def test_imbalance_options_keep_the_old_behaviour_by_default():
    for model_id in ("resnet18", "deeplabv3_resnet50"):
        model = get_model(model_id)
        assert model.param("class_weights").default == "none"
        assert model.param("loss").default == "cross_entropy"
    assert get_model("resnet18").param("loss").choices == ("cross_entropy", "focal")
    assert get_model("deeplabv3_resnet50").param("loss").choices == (
        "cross_entropy",
        "dice",
        "cross_entropy_dice",
    )
    assert get_model("resnet18").param("focal_gamma").default == 2.0
    assert "focal_gamma" not in {p.name for p in get_model("deeplabv3_resnet50").param_specs()}
    assert "class_weights" not in {p.name for p in get_model("yolov8n").param_specs()}


def test_limits_do_not_change_the_common_parameters():
    assert get_model("vit_b_16").param("image_size").max == 224
    assert next(p for p in COMMON_PARAMS if p.name == "image_size").max == 2048


def test_validation_uses_the_model_limits():
    base = {"dataset": 1, "model": "vit_b_16", "start": "transfer", "mode": "single"}
    with pytest.raises(ConfigError):
        validate(RunConfig.from_dict({**base, "params": {"image_size": 256}}), "classification")
    assert validate(RunConfig.from_dict({**base, "params": {"image_size": 224}}), "classification")


def test_yolo_offers_only_the_schedules_ultralytics_has():
    """Ultralytics has no step, one-cycle or plateau schedule: the form must not offer them (#635)."""
    for model in (m for m in MODELS if m.family == "ultralytics"):
        assert model.param("scheduler").choices == ("none", "cosine")
    assert (
        get_model("resnet18").param("scheduler").choices
        == next(p for p in COMMON_PARAMS if p.name == "scheduler").choices
    )
    base = {"dataset": 1, "model": "yolov8n", "start": "transfer", "mode": "single"}
    with pytest.raises(ConfigError) as info:
        validate(RunConfig.from_dict({**base, "params": {"scheduler": "onecycle"}}), "detection")
    assert info.value.code == "param_not_choice"
    assert validate(RunConfig.from_dict({**base, "params": {"scheduler": "cosine"}}), "detection")


def test_a_model_can_ask_for_a_multiple_and_the_check_and_the_api_say_it():
    """#635: an input size the network halves five times must be a multiple of 32; the API gives the rule to
    the form, so the dialog and the server refuse the same values."""
    spec = ParamSpec("image_size", "int", 256, 128, 1024, multiple=32)
    assert spec.check(320) == 320 and spec.as_dict()["multiple"] == 32
    with pytest.raises(CodedError) as info:
        spec.check(300)
    assert info.value.code == "param_not_multiple" and info.value.params["multiple"] == 32
    model = replace(get_model("resnet18"), multiples={"image_size": 32})
    assert model.param("image_size").multiple == 32
    assert get_model("resnet18").param("image_size").multiple is None
    assert next(p for p in model.as_dict()["params"] if p["name"] == "image_size")["multiple"] == 32


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


def test_resize_policy_defaults_keep_what_each_model_did():
    assert get_model("resnet18").param("resize").default == "stretch"
    assert get_model("deeplabv3_resnet50").param("resize").default == "stretch"
    assert get_model("fasterrcnn_resnet50_fpn").param("resize").default == "letterbox"
    assert get_model("ssdlite320_mobilenet_v3_large").param("resize").default == "stretch"
    assert get_model("resnet18").param("resize").choices == ("stretch", "letterbox", "center_crop")
    old = RunConfig.from_dict({"dataset": 1, "model": "resnet18"})  # a config made before the policy
    assert old.effective_params()["resize"] == "stretch"


def test_every_hyperparameter_has_its_name_help_and_choices_in_both_languages():
    """The dialog showed raw names ("warmup epochs", "onecycle") and English help inside the pt interface."""
    folder = Path(i18n.__file__).parent
    missing = []
    for lang in ("pt", "en"):
        texts = json.loads((folder / f"{lang}.json").read_text(encoding="utf-8"))
        for model in MODELS:
            for spec in model.param_specs():
                if spec.name == "start":  # shown as the start cards (tr.start_*)
                    continue
                keys = [f"tr.p.{spec.name}", f"tr.ph.{spec.name}"] + [
                    f"tr.opt.{c}" for c in spec.choices or ()
                ]
                missing += [f"{lang}: {key}" for key in keys if key not in texts]
    assert not sorted(set(missing))


def test_every_torchvision_classifier_is_offered_grouped_by_architecture():
    """#618: all the torchvision classification networks, each with its architecture, variant and size."""
    classifiers = [m for m in catalog.MODELS if m.task_kind == "classification" and m.family == "torchvision"]
    assert len(classifiers) == 80 and len({m.id for m in catalog.MODELS}) == len(catalog.MODELS)
    archs = {m.arch for m in classifiers}
    for arch in (
        "ResNet",
        "ResNeXt",
        "Wide ResNet",
        "DenseNet",
        "VGG",
        "MobileNet",
        "EfficientNet",
        "EfficientNetV2",
        "RegNetX",
        "RegNetY",
        "ConvNeXt",
        "ViT",
        "Swin",
        "Swin V2",
        "MaxViT",
        "ShuffleNet V2",
        "SqueezeNet",
        "MNASNet",
        "GoogLeNet",
        "Inception",
        "AlexNet",
    ):
        assert arch in archs, arch
    sizes = {m.id: m.image_size for m in classifiers}
    assert (
        sizes["tv_efficientnet_b7"],
        sizes["efficientnet_v2_s"],
        sizes["inception_v3"],
        sizes["vit_h_14"],
    ) == (
        600,
        384,
        299,
        518,
    )
    assert (
        get_model("vit_h_14").param("image_size").min == 518 == get_model("vit_h_14").param("image_size").max
    )
    assert get_model("inception_v3").param("image_size").min == 75
    assert (
        get_model("vit_l_16").param("batch_size").default == 2
        and get_model("vgg16").param("batch_size").default == 4
    )
    assert get_model("resnet18").param("batch_size").default == 8


def test_the_catalogue_describes_architecture_variant_and_size_of_each_model():
    """The dialog builds family, architecture and variant fields from these (#632)."""
    for model in catalog.MODELS:
        data = model.as_dict()
        assert data["arch"] and isinstance(data["variant"], str) and data["experimental"] is False
    resnet = get_model("resnet50").as_dict()
    assert (resnet["arch"], resnet["variant"], resnet["params_m"]) == ("ResNet", "50", 25.6)
    assert resnet["file_mb"] == pytest.approx(25.6 * 4e6 / 2**20, abs=0.1)
    assert get_model("maskrcnn_resnet50_fpn").as_dict()["arch"] == "Mask R-CNN"


def test_torchvision_and_timm_networks_of_the_same_name_have_their_own_ids():
    """timm's EfficientNet-B0 keeps its id (older runs); torchvision's is built by its own name."""
    timm_b0, tv_b0 = get_model("efficientnet_b0"), get_model("tv_efficientnet_b0")
    assert (timm_b0.family, timm_b0.library_name) == ("timm", "efficientnet_b0")
    assert (tv_b0.family, tv_b0.library_name, tv_b0.arch, tv_b0.variant) == (
        "torchvision",
        "efficientnet_b0",
        "EfficientNet",
        "B0",
    )
    assert get_model("resnet18").library_name == "resnet18"


def test_timm_offers_its_classification_families_under_its_own_ids():
    """#617: timm's families with arch and variant; a name torchvision also has gets ``timm_``."""
    timm = [m for m in catalog.MODELS if m.family == "timm" and m.task_kind == "classification"]
    assert len(timm) >= 100 and all(m.task_kind == "classification" and "timm" in m.packages for m in timm)
    for arch in (
        "ResNet",
        "MobileNetV4",
        "EfficientNet",
        "GhostNet",
        "MobileViT",
        "ConvNeXt V2",
        "NFNet",
        "ViT",
        "DeiT",
        "Swin",
        "BEiT",
        "EVA-02",
        "CaiT",
        "CoAtNet",
        "MaxViT",
        "FastViT",
    ):
        assert any(m.arch == arch for m in timm), arch
    assert get_model("efficientnet_b0").family == get_model("efficientnet_b3").family == "timm"  # older runs
    assert (get_model("timm_resnet50").library_name, get_model("resnet50").family) == (
        "resnet50",
        "torchvision",
    )
    eva = get_model("eva02_base_patch14_448").param("image_size")
    assert eva.default == eva.min == eva.max == 448  # fixed position embeddings
    fastvit = get_model("fastvit_t8").param("image_size")
    assert fastvit.default == 256 and fastvit.min < 256


def test_ultralytics_offers_three_generations_every_size_and_task():
    """#632: YOLOv8, YOLO11 and YOLO26 (the default; no YOLO12), sizes n to x, for every project type."""
    yolo = [m for m in catalog.MODELS if m.family == "ultralytics"]
    assert len(yolo) == 3 * 5 * 4 and not any("yolo12" in m.id for m in yolo)
    assert {m.arch for m in yolo} == {"YOLOv8", "YOLO11", "YOLO26"} and catalog.DEFAULT_ARCH[
        "ultralytics"
    ] == "YOLO26"
    assert {m.variant for m in yolo} == set("nsmlx")
    for model_id in ("yolo26x", "yolo26n-seg", "yolo11m-pose", "yolov8l-cls", "yolov8n", "yolo11s-seg"):
        assert get_model(model_id) is not None, (
            model_id
        )  # yolov8n and yolo11s-seg: runs saved before stay valid
    cls = get_model("yolo26s-cls")
    assert (cls.types, cls.task_kind, cls.image_size) == (("classification",), "classification", 224)
    assert (
        get_model("yolo11n").param("batch_size").default == 16
        and get_model("yolo11x").param("batch_size").default == 4
    )
    assert (
        get_model("yolo11n").as_dict()["params_m"] == 2.6
        and get_model("yolo26n").as_dict()["params_m"] == 2.4
    )


def test_every_yolo_model_shows_its_parameters_and_file_size():
    yolo = [m for m in catalog.MODELS if m.family == "ultralytics"]
    assert any(m.id.startswith("yolo26") for m in yolo)
    for m in yolo:
        d = m.as_dict()
        assert d["params_m"] is not None and d["file_mb"] is not None, m.id
    assert get_model("yolo26n").as_dict()["file_mb"] == 5.3
    assert get_model("yolo26x-pose").as_dict()["file_mb"] == 120.4
    assert get_model("yolo11n").as_dict()["file_mb"] == 9.9  # no published size: four bytes per parameter


def test_the_size_proposed_first_follows_the_device():
    assert catalog.device_size(None, False) == "n"
    assert catalog.device_size(None, True) == catalog.device_size(6144, True) == "s"
    assert catalog.device_size(8192, True) == catalog.device_size(24576, True) == "m"


def test_smp_offers_decoders_on_encoders_for_semantic_segmentation():
    """#606: the smp family, decoder as architecture and encoder as variant, installed only when used."""
    smp_models = [m for m in catalog.MODELS if m.family == "smp"]
    assert len(smp_models) == 9 * 3 + 3 and all(m.task_kind == "semantic" for m in smp_models)
    unet = get_model("smp_unet_resnet34")
    assert (unet.arch, unet.variant, unet.library_name) == ("U-Net", "ResNet-34", "Unet:resnet34")
    assert "segmentation-models-pytorch" in unet.packages and unet.image_size % 32 == 0
    assert unet.param("image_size").min == 128
    assert all(m.param("image_size").multiple == 32 for m in smp_models)  # #671: refused before training
    with pytest.raises(CodedError) as refused:
        unet.param("image_size").check(300)
    assert refused.value.code == "param_not_multiple"
    assert get_model("smp_segformer_mit_b2").library_name == "Segformer:mit_b2"
    assert "smp" in catalog.FAMILIES


def test_transformers_offers_segformer_detr_rtdetr_and_mask2former():
    """#606 step 2: Hugging Face models, architecture and variant, built from their Hub checkpoint."""
    hf = [m for m in catalog.MODELS if m.family == "transformers"]
    assert {m.arch for m in hf} == {"SegFormer (transformers)", "DETR", "RT-DETR", "Mask2Former"} and len(
        hf
    ) == 15
    assert {m.task_kind for m in hf} == {"semantic", "detection", "instance"}
    detr = get_model("hf_detr_detr_resnet_50")
    assert (detr.library_name, detr.types, "transformers" in detr.packages) == (
        "detr:facebook/detr-resnet-50",
        ("detection",),
        True,
    )
    assert detr.param("trainable_backbone_layers").min == detr.param("trainable_backbone_layers").max == 5
    assert (
        get_model("hf_segformer_mit_b0").task_kind == "semantic"
        and "other" in get_model("hf_segformer_mit_b0").notes
    )
    assert get_model("hf_mask2former_mask2former_swin_tiny_coco_instance").types == ("segmentation",)


def test_ultralytics_models_do_not_offer_a_backbone_factor():
    """#666: Ultralytics has one learning rate for the whole network; the 0.1 default made every YOLO
    fine-tune end with the yolo_backbone_lr_ignored warning."""
    for model in (m for m in MODELS if m.family == "ultralytics"):
        assert model.param("backbone_lr_factor") is None, model.id
    assert get_model("resnet18").param("backbone_lr_factor") is not None
    base = {"dataset": 1, "model": "yolov8n", "start": "finetune", "mode": "single"}
    assert validate(RunConfig.from_dict(base), "detection")
    older = validate(
        RunConfig.from_dict({**base, "params": {"backbone_lr_factor": 0.1, "epochs": 3}}), "detection"
    )
    assert older.params == {"epochs": 3}  # an older configuration still runs (Reproduce), without the factor


def test_the_two_segformer_families_say_which_they_are():
    """#671: smp and transformers both have SegFormers; the name says the family."""
    smp, hf = get_model("smp_segformer_mit_b0"), get_model("hf_segformer_mit_b0")
    assert (smp.arch, smp.label) == ("SegFormer (smp)", "SegFormer (smp) MiT-B0")
    assert (hf.arch, hf.label) == ("SegFormer (transformers)", "SegFormer (transformers) B0")


def test_mask2former_sizes_are_the_published_ones_and_smp_sizes_stay_unknown():
    """#671: published sizes only; smp publishes encoder sizes, not whole models, so those stay unknown."""
    sizes = [
        get_model(f"hf_mask2former_mask2former_swin_{s}_coco_instance").params_m
        for s in ("tiny", "small", "base")
    ]
    assert sizes == [47.5, 68.8, 107.0]
    assert all(m.params_m is None for m in catalog.MODELS if m.family == "smp")


def test_error_metrics_are_better_lower():
    """#619: the selection metrics where lower is better; the catalogue says which a model has."""
    assert catalog.lower_is_better("rmse") and catalog.lower_is_better("mae")
    assert not catalog.lower_is_better("accuracy") and not catalog.lower_is_better("r2")
    assert get_model("resnet50").as_dict()["lower_metrics"] == []


def test_the_interface_knows_the_same_error_metrics():
    """The comparison table (compare.js) marks the same metrics best when lowest, plus the loss."""
    text = (Path(catalog.__file__).parents[1] / "web" / "static" / "js" / "cv" / "compare.js").read_text(
        "utf-8"
    )
    listed = re.search(r"LOWER_IS_BETTER = \[([^\]]*)\]", text).group(1)
    assert set(re.findall(r"'(\w+)'", listed)) == set(catalog.LOWER_IS_BETTER) | {"loss"}
