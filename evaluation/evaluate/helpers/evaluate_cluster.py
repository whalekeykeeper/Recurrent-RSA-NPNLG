from ...build_test_sets import types
from typing import Union
from ...init_models import init_captioner, init_evaluator
from bayesian_agents._joint_rsa import SamplingStrategy, SpeakerType
import numpy as np


def evaluate_cluster(
    cluster: Union[types.TS1_Item, types.TS2_Item],
    strategy: SamplingStrategy = SamplingStrategy.GREEDY,
    speaker_type: SpeakerType = SpeakerType.PRAGMATIC,
    speaker_rationality: int = 5,
    n_beams: int = 10,
    cut_rate: float = 1,
    max_sentence_length: int = 60,
    max_sentences: int = 50,
) -> bool:
    """
    Evaluate a cluster.

    :param cluster: The cluster to evaluate.
    :param strategy: The sampling strategy to use.
    :param speaker_type: The type of speaker to use.
    :param speaker_rationality: The rationality of the speaker.
    :param n_beams: The number of beams to use.
    :param cut_rate: The cut rate to use.
    :param max_sentence_length: The maximum sentence length to use.
    :param max_sentences: The maximum number of sentences to use.

    :return: Whether the model correctly predicted the target image.
    """
    urls = [
        cluster["target"]["local_path"],
        *[item["local_path"] for item in cluster["distractors"]]
    ]

    captioner = init_captioner(urls)
    evaluator = init_evaluator(urls)

    pragmatic_caption = captioner.sample(
        strategy=strategy,
        speaker_type=speaker_type,
        speaker_rationality=speaker_rationality,
        n_beams=n_beams,
        cut_rate=cut_rate,
        max_sentence_length=max_sentence_length,
        max_sentences=max_sentences,
        target_image_idx=0
    )

    evaluator_posterior = evaluator.compute_posterior(
        caption=pragmatic_caption[0])

    evaluator_prediction = np.argmax(evaluator_posterior)

    if evaluator_prediction == 0:
        return True
    return False
