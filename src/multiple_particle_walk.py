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

# Particle strength thresholds for convergence detection
STRONG_THRESHOLD: float = 0.9
WEAK_THRESHOLD: float = 0.1


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


def build_undirected_knn_adjacency_list(
    data: ArrayLike, n_neighbors: int
) -> list[list[int]]:
    """Build an undirected k-nearest neighbors adjacency list.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Feature matrix.
    n_neighbors : int
        Number of neighbors for each node.

    Returns
    -------
    adjacency_list : list of list of int
        For each node, a list of its neighbor indices.
    """
    directed_graph = kneighbors_graph(
        data,
        n_neighbors=max(n_neighbors - 1, 1),
        mode='connectivity',
        include_self=False
    ).tocsr()

    undirected_graph = directed_graph.maximum(directed_graph.transpose()).tocsr()
    undirected_graph.setdiag(0)
    undirected_graph.eliminate_zeros()

    adjacency_list: list[list[int]] = []
    for i in range(undirected_graph.shape[0]):
        row_start = undirected_graph.indptr[i]
        row_end = undirected_graph.indptr[i + 1]
        adjacency_list.append(undirected_graph.indices[row_start:row_end].tolist())

    return adjacency_list


def initialize_probabilities_and_particles(
    labels: NDArray[np.intp], n_classes: int
) -> tuple[NDArray[np.floating], list[Particle]]:
    """Initialize label probabilities and create particles for labeled nodes."""
    label_probability = np.zeros((labels.size, n_classes))
    particles: list[Particle] = []

    for i in range(labels.size):
        if labels[i] == -1:
            label_probability[i, :] = 1.0 / n_classes
        else:
            particles.append(Particle(i, labels[i]))
            label_probability[i, labels[i]] = 1.0

    return label_probability, particles


def update_unlabeled_neighbor_probability(
    particle: Particle,
    neighbor: int,
    label_probability: NDArray[np.floating],
    n_classes: int,
    prob_change: float,
) -> None:
    """Update the label probabilities of an unlabeled neighbor node.

    Increases the probability of the particle's label while decreasing
    the probability of other labels. The effect is proportional to the
    particle's strength.

    Parameters
    ----------
    particle : Particle
        The particle visiting the neighbor.
    neighbor : int
        Index of the neighbor node to update.
    label_probability : ndarray of shape (n_samples, n_classes)
        Label probability matrix (modified in-place).
    n_classes : int
        Total number of classes.
    prob_change : float
        Base probability change rate.
    """
    other_label_prob_change = (particle.strength * prob_change) / (n_classes - 1)
    particle_label_prob_change = particle.strength * prob_change

    for j in range(n_classes):
        if j != particle.label:
            label_probability[neighbor, j] -= other_label_prob_change
            if label_probability[neighbor, j] < 0:
                particle_label_prob_change += label_probability[neighbor, j]
                label_probability[neighbor, j] = 0
            label_probability[neighbor, particle.label] += particle_label_prob_change


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

    for particle in particles:
        if particle.strength >= STRONG_THRESHOLD:
            strong_particles += 1
        elif particle.strength <= WEAK_THRESHOLD:
            weak_particles += 1

        neighbors = adjacency_list[particle.position]
        neighbor = neighbors[rng.integers(len(neighbors))]

        if labels[neighbor] == -1:
            update_unlabeled_neighbor_probability(
                particle, neighbor, label_probability, n_classes, prob_change
            )

        particle.strength = min(label_probability[neighbor, particle.label], 1)
        particle.position = neighbor

    return weak_particles, strong_particles


def should_stop(
    weak_particles: int,
    strong_particles: int,
    total_particles: int,
    stop_criteria: float,
) -> bool:
    """Check if the algorithm should stop based on particle convergence.

    Parameters
    ----------
    weak_particles : int
        Number of particles with strength <= WEAK_THRESHOLD.
    strong_particles : int
        Number of particles with strength >= STRONG_THRESHOLD.
    total_particles : int
        Total number of particles.
    stop_criteria : float
        Fraction of particles that must be decided (weak + strong).

    Returns
    -------
    should_stop : bool
        True if convergence criteria are met.
    """
    decided_fraction = (weak_particles + strong_particles) / total_particles
    weak_fraction = weak_particles / total_particles
    return decided_fraction > stop_criteria and weak_fraction >= (stop_criteria / 2)


def build_predictions(label_probability: NDArray[np.floating]) -> NDArray[np.intp]:
    """Build final predictions from label probabilities.

    Parameters
    ----------
    label_probability : ndarray of shape (n_samples, n_classes)
        Probability of each sample belonging to each class.

    Returns
    -------
    predictions : ndarray of shape (n_samples,)
        Predicted class label for each sample (argmax of probabilities).
    """
    return np.argmax(label_probability, axis=1).astype(int)


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

    Raises
    ------
    ValueError
        If fewer than 2 classes are present in labeled samples,
        if n_neighbors >= n_samples, or if no labeled samples exist.
    """
    data = np.asarray(data)
    labels = np.asarray(labels)
    n_samples = len(labels)

    # Input validation
    if len(data) != n_samples:
        raise ValueError(
            f"data and labels must have the same length: "
            f"got {len(data)} samples and {n_samples} labels"
        )

    if n_neighbors >= n_samples:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) must be less than n_samples ({n_samples})"
        )

    labeled_mask = labels != -1
    if not np.any(labeled_mask):
        raise ValueError("At least one labeled sample is required")

    classes = np.unique(labels[labeled_mask])
    n_classes = len(classes)

    if n_classes < 2:
        raise ValueError(
            f"At least 2 classes are required, got {n_classes}. "
            "The algorithm requires multiple classes for probability distribution."
        )

    rng = np.random.default_rng(random_state)

    # The algorithm performs better with undirected graphs
    adjacency_list = build_undirected_knn_adjacency_list(data, n_neighbors)

    # The probabilities of each instance belonging to each target class
    label_probability, particles = initialize_probabilities_and_particles(labels, n_classes)

    # Clone the particles repeated times for the main loop
    particles = particles * particle_multiplier

    # Main loop
    for _ in range(max_iter):
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
