
from .evaluate import evaluate_ts1, evaluate_ts2

from bayesian_agents._joint_rsa import SpeakerType, SamplingStrategy
import json
from .build_test_sets.vg_types import TS1, TS2
from pprint import pprint


def evaluate():
    strategy = SamplingStrategy.GREEDY
    speaker_type = SpeakerType.PRAGMATIC
    n_beams = 10
    speaker_rationality = 5

    ts1_path = "data/test_sets/ts1/ts1.json"
    ts2_path = "data/test_sets/ts2/ts2.json"

    with open(ts1_path, "r", encoding="utf-8") as f:
        ts1: TS1 = json.load(f)

    with open(ts2_path, "r", encoding="utf-8") as f:
        ts2: TS2 = json.load(f)

    print("Starting evaluation with parameters:")
    pprint(
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
