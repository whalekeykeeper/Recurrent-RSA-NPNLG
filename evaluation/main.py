from .build_test_sets import (build_ts1, build_ts2, download_vg_assets,
                              load_vg_dataset)

from .evaluate import evaluate_ts1, evaluate_ts2
from bayesian_agents._joint_rsa import SpeakerType, SamplingStrategy


def evaluate():
    download_vg_assets()
    metadata, regions, objects = load_vg_dataset()
    ts1 = build_ts1(objects=objects, regions=regions, metadata=metadata)
    ts2 = build_ts2(regions=regions, metadata=metadata)

    #  Delete unnecessary data to free up almost 4 gigs memory
    #  System might run out of ram if models are loaded before
    #  garbage collection
    del metadata
    del regions
    del objects

    strategy = SamplingStrategy.GREEDY
    speaker_type = SpeakerType.PRAGMATIC
    n_beams = 10
    speaker_rationality = 5

    print(
        "Starting evalation with parameters:",
        {
            "strategy": strategy,
            "speaker_type": speaker_type,
            "n_beams": n_beams,
            "speaker_rationality": speaker_rationality,
        }
    )

    evaluate_ts1(ts1, strategy=strategy, speaker_type=speaker_type,
                 n_beams=n_beams, speaker_rationality=speaker_rationality)
    evaluate_ts2(ts2, strategy=strategy, speaker_type=speaker_type,
                 n_beams=n_beams, speaker_rationality=speaker_rationality)
