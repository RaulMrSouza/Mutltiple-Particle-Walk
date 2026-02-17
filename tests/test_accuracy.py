import numpy as np
from sklearn import datasets, metrics

import multiple_particle_walk


def test_digits_average_accuracy_above_87_percent_over_5_runs():
    dataset = datasets.load_digits()

    rng = np.random.default_rng(42)
    random_unlabeled_points = rng.random(len(dataset.target)) < 0.9
    labels = np.copy(dataset.target)
    labels[random_unlabeled_points] = -1

    accuracies = []
    for seed in range(5):
        predictions = multiple_particle_walk.predict(dataset.data, labels, random_state=seed)
        accuracies.append(metrics.accuracy_score(dataset.target, predictions))

    mean_accuracy = float(np.mean(accuracies))
    assert mean_accuracy > 0.87, (
        f"mean_accuracy={mean_accuracy:.4f}, accuracies={[round(acc, 4) for acc in accuracies]}"
    )
