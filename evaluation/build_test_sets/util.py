import math
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from .vg_types import ImageObject, ImageRegion, Metadata
from shutil import copyfileobj

from PIL import Image as PIL_Image
import requests

import os


def visualize_image_object_and_region(image_id: int, object: ImageObject, region: ImageRegion):
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
    plt.savefig(
        f"data/vg/images/ts1_{image_id}_{object['object_id']}_{region['region_id']}.png")
    plt.cla()
    # plt.show()


def euclidean_distance(a, b):
    a_x, a_y = a
    b_x, b_y = b

    return math.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)


def download_image(image_id: int, metadata: Metadata):
    image_path = f"data/vg/images/{image_id}.jpg"

    if os.path.exists(image_path):
        return

    try:
        os.makedirs("data/vg/images")
    except:
        pass

    image_url = next(
        filter(lambda x: x["image_id"] == image_id, metadata))["url"]

    with open(image_path, "wb") as f:
        response = requests.get(image_url, stream=True)
        copyfileobj(response.raw, f)
