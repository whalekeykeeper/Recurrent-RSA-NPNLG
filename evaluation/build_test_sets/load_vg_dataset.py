import json
from .vg_types import Metadata, Objects, Regions
import os
import zipfile
from shutil import copyfileobj
import requests


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
    objects_url = "http://visualgenome.org/static/data/dataset/objects.json.zip"
    regions_url = "http://visualgenome.org/static/data/dataset/region_descriptions.json.zip"
    metadata_url = "http://visualgenome.org/static/data/dataset/image_data.json.zip"
    data_dir = os.path.join(os.getcwd(), "data", "vg")

    try:
        os.makedirs(data_dir)
    except FileExistsError:
        pass

    _download_asset(objects_url, os.path.join(data_dir, "objects"))
    _download_asset(regions_url, os.path.join(
        data_dir, "region_descriptions"))
    _download_asset(metadata_url, os.path.join(data_dir, "image_data"))


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
