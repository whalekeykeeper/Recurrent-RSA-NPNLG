from ...build_test_sets import types
from typing import Union
from ...init_models import init_captioner, init_evaluator
from bayesian_agents._joint_rsa import SamplingStrategy, SpeakerType
import numpy as np


def evaluate_cluster(cluster: Union[types.TS1_Item, types.TS2_Item]) -> bool:
    urls = [
        cluster["target"]["local_path"],
        *[item["local_path"] for item in cluster["distractors"]]
    ]

    captioner = init_captioner(urls)
    evaluator = init_evaluator(urls)

    pragmatic_caption = captioner.sample(
        strategy=SamplingStrategy.BEAM,
        speaker_type=SpeakerType.PRAGMATIC,
        speaker_rationality=5,
        n_beams=10,
        cut_rate=1,
        max_sentence_length=60,
        max_sentences=50,
        target_image_idx=0
    )

    print("Pragmatic caption:\n", pragmatic_caption)
    evaluator_posterior = evaluator.compute_posterior(
        caption=pragmatic_caption[0])

    evaluator_prediction = np.argmax(evaluator_posterior)
    print("evaluator prediction", evaluator_prediction)

    if evaluator_prediction == 0:
        return True
    return False
