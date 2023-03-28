from ...build_test_sets import types
from ..helpers.evaluate_cluster import evaluate_cluster
from tqdm import tqdm
from bayesian_agents._joint_rsa import SamplingStrategy, SpeakerType


def evaluate_ts1(
    test_set: types.TS1,
    strategy: SamplingStrategy = SamplingStrategy.GREEDY,
    speaker_type: SpeakerType = SpeakerType.PRAGMATIC,
    speaker_rationality: int = 5,
    n_beams: int = 10,
    cut_rate: float = 1,
    max_sentence_length: int = 60,
    max_sentences: int = 50,
):
    correctly_predicted = 0
    total = 0
    for cluster in tqdm(test_set):
        is_correct_prediction = evaluate_cluster(
            cluster,
            strategy=strategy,
            speaker_type=speaker_type,
            speaker_rationality=speaker_rationality,
            n_beams=n_beams,
            cut_rate=cut_rate,
            max_sentence_length=max_sentence_length,
            max_sentences=max_sentences,
        )
        if (is_correct_prediction):
            correctly_predicted += 1
        total += 1

    print("Accuracy on TS1", correctly_predicted / total)

    return correctly_predicted, total
