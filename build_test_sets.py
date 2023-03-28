
from evaluation.build_test_sets import (build_ts1, build_ts2,
                                        download_vg_assets, load_vg_dataset)


def main():
    download_vg_assets()
    metadata, regions, objects = load_vg_dataset()
    build_ts1(objects=objects, regions=regions, metadata=metadata)
    build_ts2(regions=regions, metadata=metadata)


if __name__ == '__main__':
    main()
