from .build_test_sets import load_vg_dataset, build_ts1, download_vg_assets


def evaluate():
    download_vg_assets()
    metadata, regions, objects = load_vg_dataset()
    build_ts1(objects=objects, regions=regions, metadata=metadata)
