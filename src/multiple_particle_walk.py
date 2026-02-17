# -*- coding: utf-8 -*-
"""
Multiple Particle Walk - Semi-supervised Classification

Author: Raul M. Souza

The Multiple Particle Competition and Cooperation or Multiple Particle Walk algorithm
is a semi-supervised classification model.

It is a modified version of the original Particle Competition and Cooperation algorithm,
see the reference for more information about the original.

The difference in this version is that each labeled sample will generate multiple 
particles instead of one and there is only random walk movement.

References
----------
[1] Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gonçalves; Pedrycz, Witold; Liu, Jiming, 
    "Particle Competition and Cooperation in Networks for Semi-Supervised Learning," 
    Knowledge and Data Engineering, IEEE Transactions on, vol.24, no.9, pp.1686-1698, Sept. 2012
    doi: 10.1109/TKDE.2011.119
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.neighbors import kneighbors_graph


class Particle:
    """A particle that walks on the graph and updates node probabilities.
    
    Attributes
    ----------
    strength : float
        Current strength of the particle (0.0 to 1.0).
    position : int
        Current node index in the graph.
    label : int
        Class label this particle promotes.
    """
    
    def __init__(self, position: int, label: int) -> None:
        self.strength: float = 1.0
        self.position: int = position
        self.label: int = label


def build_undirected_knn_adjacency_list(data, n_neighbors):
    directed_graph = kneighbors_graph(
        data,
        n_neighbors=max(n_neighbors - 1, 1),
        mode='connectivity',
        include_self=False
    ).tocsr()

    undirected_graph = directed_graph.maximum(directed_graph.transpose()).tocsr()
    undirected_graph.setdiag(0)
    undirected_graph.eliminate_zeros()

    graph = []
    for i in range(0, undirected_graph.shape[0]):
        row_start = undirected_graph.indptr[i]
        row_end = undirected_graph.indptr[i + 1]
        graph.append(undirected_graph.indices[row_start:row_end].tolist())

    return graph


def initialize_probabilities_and_particles(
    labels: NDArray[np.intp], n_classes: int
) -> tuple[NDArray[np.floating], list[Particle]]:
    """Initialize label probabilities and create particles for labeled nodes."""
    label_probability = np.zeros((labels.size, n_classes))
    particles: list[Particle] = []

    for i in range(0, labels.size):
        if labels[i] == -1:
            for j in range(0, n_classes):
                label_probability[i, j] = 1.0 / n_classes
        else:
            particles.append(Particle(i, labels[i]))
            label_probability[i, labels[i]] = 1.0

    return label_probability, particles


def update_unlabeled_neighbor_probability(particle_item, neighbor, label_probability, n_classes, prob_change):
    other_label_prob_change = (particle_item.strength * prob_change)/(n_classes-1)
    particle_label_prob_change = particle_item.strength * prob_change

    for j in range(0,n_classes):
        if j != particle_item.label:
            label_probability[neighbor,j] -= other_label_prob_change
            if label_probability[neighbor,j] < 0:
                particle_label_prob_change += label_probability[neighbor,j]
                label_probability[neighbor,j] = 0
            label_probability[neighbor,particle_item.label] += particle_label_prob_change


def run_main_iteration(
    particles: list[Particle],
    adjacency_list: list[list[int]],
    labels: NDArray[np.intp],
    label_probability: NDArray[np.floating],
    n_classes: int,
    prob_change: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Execute one iteration of the particle walk algorithm."""
    weak_particles = 0
    strong_particles = 0

    for i in range(0, len(particles)):
        particle_item = particles[i]

        if particle_item.strength >= 0.9:
            strong_particles += 1
        elif particle_item.strength <= 0.1:
            weak_particles += 1

        neighbors = adjacency_list[particle_item.position]
        neighbor = neighbors[rng.integers(len(neighbors))]

        if labels[neighbor] == -1:
            update_unlabeled_neighbor_probability(
                particle_item, neighbor, label_probability, n_classes, prob_change
            )

        particle_item.strength = min(label_probability[neighbor, particle_item.label], 1)
        particle_item.position = neighbor

    return weak_particles, strong_particles


def should_stop(weak_particles, strong_particles, total_particles, stop_criteria):
    return (weak_particles + strong_particles)>((total_particles)*stop_criteria) and weak_particles>=(total_particles*(stop_criteria/2))


def build_predictions(label_probability):
    predictions = np.zeros(len(label_probability)).astype(int)

    for i in range(0,len(label_probability)):
        predictions[i] = np.argmax(label_probability[i])

    return predictions


def predict(
    data: ArrayLike,
    labels: ArrayLike,
    n_neighbors: int = 6,
    particle_multiplier: int = 30,
    prob_change: float = 0.1,
    stop_criteria: float = 0.3,
    max_iter: int = 100,
    random_state: Optional[int] = None,
) -> NDArray[np.intp]:
    """Predict labels for unlabeled samples using the Multiple Particle Walk algorithm.
    
    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Feature matrix.
    labels : array-like of shape (n_samples,)
        Label array where -1 indicates unlabeled samples.
    n_neighbors : int, default=6
        Number of neighbors for k-NN graph construction.
    particle_multiplier : int, default=30
        Number of particles to spawn per labeled sample.
    prob_change : float, default=0.1
        Probability change rate when particles visit nodes.
    stop_criteria : float, default=0.3
        Fraction of particles that must be weak/strong to stop.
    max_iter : int, default=100
        Maximum number of iterations.
    random_state : int or None, default=None
        Seed for reproducible results. If None, results may vary between runs.
    
    Returns
    -------
    predictions : ndarray of shape (n_samples,)
        Predicted class labels for all samples.
    
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> data, target = load_iris(return_X_y=True)
    >>> labels = target.copy()
    >>> labels[::2] = -1  # Mark half as unlabeled
    >>> predictions = predict(data, labels, random_state=42)
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(random_state)
    
    classes = np.unique(labels)
    classes = classes[classes != -1]
    n_classes = len(classes)

    # The algorithm performs better with undirected graphs
    adjacency_list = build_undirected_knn_adjacency_list(data, n_neighbors)

    # The probabilities of each instance belonging to each target class
    label_probability, particles = initialize_probabilities_and_particles(labels, n_classes)

    # Clone the particles repeated times for the main loop
    particles = particles * particle_multiplier

    # Main loop
    for _ in range(0, max_iter):
        weak_particles, strong_particles = run_main_iteration(
            particles,
            adjacency_list,
            labels,
            label_probability,
            n_classes,
            prob_change,
            rng,
        )
            
        # Check the stop criteria    
        if should_stop(weak_particles, strong_particles, len(particles), stop_criteria):
            break

    return build_predictions(label_probability)
