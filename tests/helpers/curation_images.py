"""Images for the curation tests: a small project of generated pictures with known defects."""

from PIL import Image
from sample_images import make_image

from activelabel.curation.issues import Thresholds
from activelabel.images.manifest import ImageRecord, Manifest
from activelabel.jobs import Cancelled

LIMITS = Thresholds(min_side=40, low_contrast=3.0)  # the sample images are small and soft


def add(manifest, image_id, **kwargs):
    path = manifest.project_dir / "images" / f"{image_id}.png"
    make_image(path, **kwargs)
    with Image.open(path) as img:
        width, height = img.size
    record = ImageRecord(
        image_id, f"{image_id}.png", f"images/{image_id}.png", width, height, 1, "2026-01-01"
    )
    manifest.add([record])
    return path


def build(tmp_path):
    manifest = Manifest(tmp_path)
    add(manifest, "ok1", seed=1)
    add(manifest, "ok2", seed=2)
    add(manifest, "ok3", seed=3)
    add(manifest, "dark", seed=4, brightness=10)
    add(manifest, "blur", seed=5, blur=3)
    add(manifest, "tiny", size=(32, 24), seed=6)
    add(manifest, "wide", size=(200, 48), seed=7)
    return manifest


class Stop:
    def __init__(self, after):
        self.calls, self.after = 0, after

    def update(self, done=None, total=None, message=None):
        self.calls += 1
        if self.calls > self.after:
            raise Cancelled

    def check(self):
        pass
