# -*- coding: utf-8 -*-
"""Multiple Particle Walk Example.

Author: Raul M. Souza
"""

import numpy as np
from sklearn import datasets, metrics

import multiple_particle_walk

dataset = datasets.load_digits()

# Take a random labeled sample from the dataset
rng = np.random.default_rng(42)
random_unlabeled_points = rng.random(len(dataset.target)) < 0.9
labels = np.copy(dataset.target)
labels[random_unlabeled_points] = -1

predictions = multiple_particle_walk.predict(dataset.data, labels, random_state=42)

print(f"\nAccuracy: {metrics.accuracy_score(dataset.target, predictions) * 100:.2f}%")
