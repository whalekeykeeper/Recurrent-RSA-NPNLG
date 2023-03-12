import math
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from .vg_types import ImageObject, ImageRegion, Metadata, Regions, TS1_Raw_Cluster, TS1_Item, ImageDef, TS2_Raw_Cluster
from shutil import copyfileobj
from typing import List
from itertools import combinations
from functools import cache
from PIL import Image as PIL_Image
import requests
import os


def visualize_image_object_and_region(image_id: int, object: ImageObject, region: ImageRegion, save_path: str = None):
    image_path = f"data/vg/images/{image_id}.jpg"

    with open(image_path, "rb") as f:
        image = PIL_Image.open(f)
        plt.imshow(image)
    ax = plt.gca()
    ax.add_patch(
        Rectangle(
            (object["x"], object["y"]),
            object["w"],
            object["h"],
            fill=False,
            edgecolor="red",
            linewidth=3
        )
    )
    ax.add_patch(
        Rectangle(
            (region["x"], region["y"]),
            region["width"],
            region["height"],
            fill=False,
            edgecolor="yellow",
            linewidth=3
        ))
    ax.text(region["x"], region["y"], region["phrase"], style='italic', bbox={
            'facecolor': 'white', 'alpha': 0.7, 'pad': 5})

    fig = plt.gcf()
    plt.tick_params(labelbottom='off', labelleft='off')
    if save_path is not None:
        plt.savefig(
            save_path)

    plt.cla()
    # plt.show()


def euclidean_distance(a, b):
    a_x, a_y = a
    b_x, b_y = b

    return math.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)


def get_object_bounding_region(image_id: int, object: ImageObject, regions: Regions):
    """
    Given an object in an image in the Visual Genome dataset, find the
    region of that image that most closely fits the object.
    """

    same_image_regions = next(
        filter(lambda x: x["id"] == image_id, regions))["regions"]

    object_a = (object["x"], object["y"])
    object_b = (object["x"] + object["w"], object["y"] + object["h"])

    def _sorter(x: ImageRegion):
        # Sort the regions of the image in such a way that the best
        # bounding region for the object is first.

        region_a = (x["x"], x["y"])
        region_b = (x["x"] + x["width"], x["y"] + x["height"])

        distance_a = euclidean_distance(object_a, region_a)
        distance_b = euclidean_distance(object_b, region_b)

        return distance_a + distance_b

    return min(same_image_regions, key=_sorter)


def visualize_ts1_cluster(cluster: TS1_Raw_Cluster, regions: Regions, object_type: str = None, cluster_id: int = None):
    for image_id, object in cluster:
        data_dir = os.path.join(os.getcwd(), "data", "vg",
                                "object_region_visualizations", object_type, str(cluster_id))
        try:
            os.makedirs(data_dir)
        except FileExistsError:
            pass

        region = get_object_bounding_region(image_id, object, regions)

        out_path = os.path.join(
            data_dir, f"{image_id}_{object['object_id']}_{region['region_id']}.png")
        visualize_image_object_and_region(
            image_id, object, region, save_path=out_path)


def object_satisifies_minimum_dimensions(object: ImageObject, min_dims: int = 100):
    return object["w"] > min_dims and object["h"] > min_dims


def region_satisfies_minimum_dimensions(region: ImageRegion, min_dims: int = 100):
    return region["width"] > min_dims and region["height"] > min_dims


def download_image(image_id: int, metadata: Metadata) -> str:
    image_path = f"data/vg/images/{image_id}.jpg"
    image_url = next(
        filter(lambda x: x["image_id"] == image_id, metadata))["url"]

    if os.path.exists(image_path):
        return image_path, image_url

    try:
        os.makedirs("data/vg/images")
    except:
        pass

    with open(image_path, "wb") as f:
        response = requests.get(image_url, stream=True)
        copyfileobj(response.raw, f)

    return image_path, image_url


def process_ts1_cluster(cluster: TS1_Raw_Cluster, metadata: Metadata = None, object_type: str = "", cluster_id: int = 1) -> TS1_Item:

    print(
        f"Processing TS1 cluster {cluster_id} for object type '{object_type}'")

    data_dir = os.path.join(os.getcwd(), "data", "test_sets",
                            "ts1", object_type, str(cluster_id))

    try:
        os.makedirs(data_dir)
    except FileExistsError:
        pass

    images: List[ImageDef] = []

    for image_id, object in cluster:
        full_image_local_path, remote_url = download_image(image_id, metadata)
        output_image_name = f"{image_id}_{object['object_id']}.jpg"
        cropped_image_path = os.path.join(
            data_dir,
            output_image_name
        )
        with open(full_image_local_path, "rb") as f:
            image = PIL_Image.open(f)
            crop_definition = (object["x"], object["y"], object["x"] + object["w"],
                               object["y"] + object["h"])
            cropped = image.crop(crop_definition)
            cropped.save(cropped_image_path)
            images.append({
                "local_path": os.path.join("data", "test_sets", "ts1", object_type, str(cluster_id), output_image_name),
                "original_image_id": image_id,
                "original_object_id": object["object_id"],
                "remote_url": remote_url
            })

    ts1_item: TS1_Item = {
        "target": images[0],
        "distractors": images[1:],
        "cluster_id": cluster_id,
        "object_type": object_type,
    }

    return ts1_item


def visualize_image_region(image_id: int, region: ImageRegion, save_path: str = None):
    """
    Visualize a region of an image in the Visual Genome dataset
    """

    image_path = f"data/vg/images/{image_id}.jpg"

    with open(image_path, "rb") as f:
        image = PIL_Image.open(f)
        plt.imshow(image)

    ax = plt.gca()
    ax.add_patch(
        Rectangle(
            (region["x"], region["y"]),
            region["width"],
            region["height"],
            fill=False,
            edgecolor="yellow",
            linewidth=3
        ))
    ax.text(region["x"], region["y"], region["phrase"], style='italic', bbox={
            'facecolor': 'white', 'alpha': 0.7, 'pad': 5})

    plt.tick_params(labelbottom='off', labelleft='off')
    if save_path is not None:
        plt.savefig(
            save_path)

    plt.cla()
    # plt.show()


def avg_word_overlap(cluster: TS2_Raw_Cluster) -> float:
    """
    Compute average word overlap in a cluster of image 
    regions of the VG dataset
    """
    phrases = [x[1]["phrase"] for x in cluster]
    sets = [set(x.split()) for x in phrases]
    intersections = sets[0]
    for s in sets[1:]:
        intersections = intersections.intersection(s)
    return len(intersections)


def visualize_ts2_cluster(cluster: TS2_Raw_Cluster, cluster_id: int = None):
    for image_id, region in cluster:
        data_dir = os.path.join(os.getcwd(), "data", "vg",
                                "region_visualizations", str(cluster_id))
        try:
            os.makedirs(data_dir)
        except FileExistsError:
            pass

        out_path = os.path.join(
            data_dir, f"{image_id}_{region['region_id']}.png")
        visualize_image_region(
            image_id, region, save_path=out_path)
