from ...build_test_sets import types
from ..helpers.evaluate_cluster import evaluate_cluster
from tqdm import tqdm


def evaluate_ts1(test_set: types.TS1):
    correctly_predicted = 0
    total = 0
    for cluster in tqdm(test_set):
        is_correct_prediction = evaluate_cluster(cluster)
        if (is_correct_prediction):
            correctly_predicted += 1
        total += 1

    print("Accuracy on TS1", correctly_predicted / total)

    return correctly_predicted, total
