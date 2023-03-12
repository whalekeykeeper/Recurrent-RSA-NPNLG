import random
import json

from .util import (process_cluster,
                   satisifies_minimum_dimensions,
                   visualize_cluster
                   )
from .vg_types import (TS1, TS1_Item, TS2, Metadata, Objects,
                       Regions, TS1_Raw_Clusters, TS1_Raw_Cluster)


def _get_object_clusters(object_type: str = "person", n_cluster_items: int = 10, n_clusters: int = 10, objects: Objects = None) -> TS1_Raw_Clusters:
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
            if object_type in object["names"] and satisifies_minimum_dimensions(object):
                same_type_objects.append((image_id, object))

    random.shuffle(same_type_objects)

    clusters = []

    for i in range(n_clusters):
        items = same_type_objects[i *
                                  n_cluster_items: (i + 1) * n_cluster_items]
        clusters.append(items)

    return clusters


def build_ts1(objects: Objects = None, regions: Regions = None, metadata: Metadata = None) -> TS1:

    target_objects_types = [
        "man",
        "person",
        "woman",
        "building",
        "sign",
        "table",
        "bus",
        "window",
        "sky",
        "tree"
    ]

    test_set: TS1 = []

    for object_type in target_objects_types:

        clusters = _get_object_clusters(
            object_type=object_type, n_cluster_items=10, n_clusters=10, objects=objects)

        for i, cluster in enumerate(clusters):
            processed_cluster: TS1_Item = process_cluster(
                cluster, metadata=metadata, object_type=object_type, cluster_id=i)
            test_set.append(processed_cluster)

            # Optionally, visualize the objects and regions in the cluster.
            # Images will be saved to the data/vg/object_region_visualizations folder.
            # It is important to call this after process_cluster as otherwise
            # the images are not downloaded.

            visualize_cluster(cluster, regions=regions,
                              object_type=object_type, cluster_id=i)

    with open("data/test_sets/ts1/ts1.json", "w") as f:
        json.dump(test_set, f, indent=4)

    return test_set


def build_ts2() -> TS2:
    pass
