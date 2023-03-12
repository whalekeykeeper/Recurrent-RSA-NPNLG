from .vg_types import TS1, TS2, Objects, Regions, ImageObject, ImageRegion, Metadata
from .util import euclidean_distance, visualize_image_object_and_region, download_image
import random
from typing import Tuple, List


def _get_random_objects(object_type: str = "person", n: int = 10, objects: Objects = None) -> List[Tuple[int, ImageObject]]:
    """
    Get a random sample of objects of the given type.
    """

    if not objects:
        raise ValueError("No objects provided.")

    same_type_objects = []

    for image in objects:
        image_id = image["image_id"]
        image_objects = image["objects"]
        for object in image_objects:
            if object_type in object["names"]:
                same_type_objects.append((image_id, object))

    return random.sample(same_type_objects, n)


def _get_object_bounding_region(image_id: int, object: ImageObject, regions: Regions):
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


def build_ts1(objects: Objects = None, regions: Regions = None, metadata: Metadata = None) -> TS1:

    target_objects = [
        "man",
        # "person",
        # "woman",
        # "building",
        # "sign",
        # "table",
        # "bus",
        # "window",
        # "sky",
        # "tree"
    ]

    objects = _get_random_objects(object_type="person", n=10, objects=objects)

    for image_id, object in objects:
        download_image(image_id, metadata)
        region = _get_object_bounding_region(image_id, object, regions)
        visualize_image_object_and_region(image_id, object, region)


def build_ts2() -> TS2:
    pass
