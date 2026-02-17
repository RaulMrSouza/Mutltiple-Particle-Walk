import random
import sys
from pathlib import Path

import numpy as np
from sklearn import datasets
from sklearn import metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import multiple_particle_walk


def test_digits_average_accuracy_above_87_percent_over_5_runs():
    dataset = datasets.load_digits()

    rng = np.random.default_rng(42)
    random_unlabeled_points = rng.random(len(dataset.target)) < 0.9
    labels = np.copy(dataset.target)
    labels[random_unlabeled_points] = -1

    accuracies = []
    for seed in range(5):
        random.seed(seed)
        predictions = multiple_particle_walk.predict(dataset.data, labels)
        accuracies.append(metrics.accuracy_score(dataset.target, predictions))

    mean_accuracy = float(np.mean(accuracies))
    assert mean_accuracy > 0.87, (
        f"mean_accuracy={mean_accuracy:.4f}, accuracies={[round(acc, 4) for acc in accuracies]}"
    )
