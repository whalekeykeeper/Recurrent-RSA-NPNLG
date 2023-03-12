from typing import List, TypedDict

# Metadata types


class MetadataItem(TypedDict):
    image_id: int
    url: str
    width: int
    height: int
    flickr_id: int
    coco_id: str


Metadata = List[MetadataItem]


# Object types


class ImageObject(TypedDict):
    synsets: List[str]
    object_id: int
    merged_object_ids: List[int]
    names: List[str]
    h: int
    w: int
    x: int
    y: int


class ImageObjects(TypedDict):
    image_id: int
    objects: List[ImageObject]


Objects = List[ImageObjects]

# Region types


class ImageRegion(TypedDict):
    image_id: int
    region_id: int
    phrase: str
    x: int
    y: int
    width: int
    height: int


class ImageRegions(TypedDict):
    id: int
    regions: List[ImageRegion]


Regions = List[ImageRegions]


# Test set types

class TS1_Item (TypedDict):
    target: str
    distractors: List[str]


class TS2_Item (TypedDict):
    target: str
    distractors: List[str]


TS1 = List[TS1_Item]
TS2 = List[TS2_Item]
