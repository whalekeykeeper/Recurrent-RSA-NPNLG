import json
from .vg_types import Metadata, Objects, Regions, ImageRegion, Attributes, ImageObject
import os
import zipfile
from shutil import copyfileobj
import requests
from typing import Dict, List


def _download_asset(url: str, outpath: str):
    final_zip_path = outpath + ".zip"
    final_path = outpath + ".json"

    if os.path.exists(final_path):
        return

    print("Downloading", url, "to", outpath)

    with open(final_zip_path, "wb") as f:
        r = requests.get(url, stream=True)
        copyfileobj(r.raw, f)
    with zipfile.ZipFile(final_zip_path, "r") as zip_ref:
        zip_ref.extractall(outpath)

    for f in os.listdir(outpath):
        current_path = os.path.join(outpath, f)
        os.rename(current_path, final_path)

    os.remove(final_zip_path)
    os.rmdir(outpath)


def download_vg_assets():
    objects_url = "http://visualgenome.org/static/data/dataset/image_data_v1.json.zip"
    regions_url = "http://visualgenome.org/static/data/dataset/region_descriptions_v1.json.zip"
    metadata_url = "http://visualgenome.org/static/data/dataset/image_data.json.zip"
    attributes_url = "http://visualgenome.org/static/data/dataset/attributes_v1.json.zip"
    data_dir = os.path.join(os.getcwd(), "data", "vg")

    try:
        os.makedirs(data_dir)
    except FileExistsError:
        pass

    _download_asset(objects_url, os.path.join(data_dir, "objects"))
    _download_asset(regions_url, os.path.join(
        data_dir, "region_descriptions"))
    _download_asset(metadata_url, os.path.join(data_dir, "image_data"))
    _download_asset(attributes_url, os.path.join(data_dir, "attributes"))


def _load_regions() -> Regions:
    with open("data/vg/region_descriptions.json", "r") as f:
        region_descriptions = json.load(f)
    d: Dict[int, List[ImageRegion]] = {}
    for definition in region_descriptions:
        image_id = definition["id"]
        regions = definition["regions"]
        if image_id not in d:
            d[image_id] = []
        for region in regions:
            d[image_id].append({
                "image_id": image_id,
                "region_id": region["id"],
                "x": region["x"],
                "y": region["y"],
                "height": region["height"],
                "width": region["width"],
                "phrase": region["phrase"],
            })
    r: Regions = []
    for key, value in d.items():
        r.append({
            "id": key,
            "regions": value
        })
    return r


def _load_objects(attributes: Attributes) -> Objects:
    d: Dict[int, List[ImageObject]] = {}
    for definition in attributes:
        image_id = definition["id"]
        objects = definition["attributes"]
        if not image_id in d:
            d[image_id] = []
        for obj in objects:
            d[image_id].append({
                "object_id": obj["id"],
                "merged_object_ids": [],
                "names": obj["object_names"],
                "synsets": [],
                "h": obj["h"],
                "w": obj["w"],
                "x": obj["x"],
                "y": obj["y"],
            })
    o: Objects = []
    for key, value in d.items():
        o.append({
            "image_id": key,
            "objects": value
        })
    return o


def _load_metadata() -> Metadata:
    with open("data/vg/image_data.json", "r") as f:
        metadata = json.load(f)
    return metadata


def _load_attributes() -> Attributes:
    with open("data/vg/attributes.json", "r") as f:
        attributes = json.load(f)
    return attributes


def load_vg_dataset():
    attributes = _load_attributes()
    regions = _load_regions()
    objects = _load_objects(attributes)
    metadata = _load_metadata()

    return metadata, regions, objects
