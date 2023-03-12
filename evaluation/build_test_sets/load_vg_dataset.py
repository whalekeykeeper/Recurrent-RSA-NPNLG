import json
from .vg_types import Metadata, Objects, Regions


def download_vg_assets():
    pass


def _load_regions() -> Regions:
    with open("data/vg/region_descriptions.json", "r") as f:
        region_descriptions = json.load(f)
    return region_descriptions


def _load_objects() -> Objects:
    with open("data/vg/objects.json", "r") as f:
        object_descriptions = json.load(f)
    return object_descriptions


def _load_metadata() -> Metadata:
    with open("data/vg/image_data.json", "r") as f:
        metadata = json.load(f)
    return metadata


def load_vg_dataset():
    regions = _load_regions()
    objects = _load_objects()
    metadata = _load_metadata()

    return metadata, regions, objects
