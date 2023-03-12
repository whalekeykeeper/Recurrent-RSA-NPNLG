from ...build_test_sets import types
from ..helpers.evaluate_cluster import evaluate_cluster


def evaluate_ts1(test_set: types.TS1):
    correctly_predicted = 0
    total = 0
    for cluster in test_set:
        is_correct_prediction = evaluate_cluster(cluster)
        if (is_correct_prediction):
            correctly_predicted += 1
        total += 1
        print(f"Evaluated TS1 / cluster {cluster['cluster_id']}")
        print(
            f"Accuracy so far: {correctly_predicted / total} ({correctly_predicted} / {total})")
        print("\n-----------------------\n")

    return correctly_predicted, total
