import json
import random
from tqdm import tqdm
from itertools import product
from typing import List, MutableSet, Tuple

from .util import (avg_word_overlap, download_image,
                   object_satisifies_minimum_dimensions, process_ts1_cluster,
                   region_satisfies_minimum_dimensions, visualize_ts1_cluster,
                   visualize_ts2_cluster)
from .vg_types import (TS1, TS2, ImageRegion, Metadata, Objects, Regions,
                       TS1_Item, TS1_Raw_Clusters, TS2_Raw_Cluster,
                       TS2_Raw_Clusters)


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
            if object_type in object["names"] and object_satisifies_minimum_dimensions(object):
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
            processed_cluster: TS1_Item = process_ts1_cluster(
                cluster, metadata=metadata, object_type=object_type, cluster_id=i)
            test_set.append(processed_cluster)

            # Optionally, visualize the objects and regions in the cluster.
            # Images will be saved to the data/vg/object_region_visualizations folder.
            # It is important to call this after process_cluster as otherwise
            # the images are not downloaded.

            visualize_ts1_cluster(cluster, regions=regions,
                                  object_type=object_type, cluster_id=i)

    with open("data/test_sets/ts1/ts1.json", "w") as f:
        json.dump(test_set, f, indent=4)

    return test_set


def _acceptable_next_cluster_item(cluster: TS2_Raw_Cluster, exhausted_ids: MutableSet[int]):
    def inner(item: Tuple[int, ImageRegion]) -> bool:
        _, region = item
        if region["region_id"] in exhausted_ids:
            return -1
        return avg_word_overlap([*cluster, item])

    return inner


def _get_region_clusters(regions: Regions = None, n_cluster_items: int = 10, n_clusters: int = 10) -> TS2_Raw_Clusters:
    """
    Get a random sample of regions.
    """

    flattened_regions: List[Tuple[int, ImageRegion]] = []

    for image in regions:
        image_id = image["id"]
        image_regions = image["regions"]
        for region in image_regions:
            if region_satisfies_minimum_dimensions(region):
                flattened_regions.append((image_id, region))

    clusters: TS2_Raw_Clusters = [[item] for item in random.sample(
        flattened_regions, n_clusters * 2)]

    exhausted_ids: MutableSet[int] = set(
        [item[1]["region_id"] for cluster in clusters for item in cluster])

    operations = list(product(list(range(n_cluster_items - 1)),
                              list(range(len(clusters)))))

    for _, cluster_idx in tqdm(operations):
        cluster = clusters[cluster_idx]
        next_item = max(flattened_regions,
                        key=_acceptable_next_cluster_item(cluster, exhausted_ids))
        cluster.append(next_item)
        exhausted_ids.add(next_item[1]["region_id"])

    clusters.sort(key=lambda cluster: avg_word_overlap(
        cluster), reverse=True)
    with open("ts2_clusters_full.json", "w") as f:
        json.dump(clusters, f, indent=4)
    return clusters[:n_clusters]


def build_ts2(regions: Regions = None, metadata: Metadata = None) -> TS2:
    clusters = _get_region_clusters(
        regions=regions, n_cluster_items=10, n_clusters=100)
    with open("ts2_clusters.json", "w") as f:
        json.dump(clusters, f, indent=4)
    # for i, cluster in enumerate(clusters):
    #     for image_id, _ in cluster:
    #         download_image(image_id, metadata=metadata)
    #     visualize_ts2_cluster(cluster, cluster_id=i)
