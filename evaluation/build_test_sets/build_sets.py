import random
import json

from .util import (process_ts1_cluster,
                   satisifies_minimum_dimensions,
                   visualize_ts1_cluster,
                   avg_word_overlap,
                   visualize_ts2_cluster,
                   download_image
                   )
from .vg_types import (TS1, TS1_Item, TS2, Metadata, Objects,
                       Regions, TS1_Raw_Clusters, TS2_Raw_Clusters, TS2_Raw_Cluster, ImageRegion)

from typing import MutableSet, List, Tuple


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


def _acceptable_next_cluster_item(cluster: TS2_Raw_Cluster, exhausted_ids: MutableSet[int], minimum_target_overlap: int):
    def inner(item: Tuple[int, ImageRegion]) -> bool:
        _, region = item
        if region["region_id"] in exhausted_ids:
            return False
        overlap = avg_word_overlap([*cluster, item])
        return overlap >= minimum_target_overlap

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
            flattened_regions.append((image_id, region))

    clusters: TS2_Raw_Clusters = []
    exhausted_ids: MutableSet[int] = set()
    minimum_target_overlap = 5

    while len(clusters) < n_clusters:
        print(len(clusters))
        if len(clusters) % 10 == 0:
            with open("ts2_clusters.json", "w") as f:
                json.dump(clusters, f, indent=4)

        first_image_id, first_region = random.choice(flattened_regions)

        if first_region["region_id"] in exhausted_ids:
            continue

        cluster: TS2_Raw_Cluster = [(first_image_id, first_region)]
        exhausted_ids.add(first_region["region_id"])
        for _ in range(n_cluster_items - 1):
            try:
                next_image_id, next_region = next(
                    filter(
                        _acceptable_next_cluster_item(
                            cluster, exhausted_ids, minimum_target_overlap),
                        flattened_regions
                    )
                )
                exhausted_ids.add(next_region["region_id"])
                cluster.append((next_image_id, next_region))
            except StopIteration:
                break

        overlap = avg_word_overlap(cluster)
        if overlap >= minimum_target_overlap and len(cluster) == n_cluster_items:
            clusters.append(cluster)
        else:
            for _, region in cluster:
                exhausted_ids.remove(region["region_id"])

    return clusters


def build_ts2(regions: Regions = None, metadata: Metadata = None) -> TS2:
    clusters = _get_region_clusters(
        regions=regions, n_cluster_items=10, n_clusters=10)
    with open("ts2_clusters.json", "w") as f:
        json.dump(clusters, f, indent=4)
    # for i, cluster in enumerate(clusters):
    #     for image_id, _ in cluster:
    #         download_image(image_id, metadata=metadata)
    #     visualize_ts2_cluster(cluster, cluster_id=i)
