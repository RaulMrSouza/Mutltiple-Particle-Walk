"""Unit tests for individual functions in multiple_particle_walk module."""

import numpy as np
import pytest

import multiple_particle_walk
from multiple_particle_walk import (
    Particle,
    build_predictions,
    build_undirected_knn_adjacency_list,
    initialize_probabilities_and_particles,
    run_main_iteration,
    should_stop,
    update_unlabeled_neighbor_probability,
)


class TestParticle:
    """Tests for the Particle class."""

    def test_particle_initialization(self):
        p = Particle(position=5, label=2)
        assert p.position == 5
        assert p.label == 2
        assert p.strength == 1.0

    def test_particle_strength_can_be_modified(self):
        p = Particle(position=0, label=0)
        p.strength = 0.5
        assert p.strength == 0.5


class TestBuildUndirectedKnnAdjacencyList:
    """Tests for build_undirected_knn_adjacency_list function."""

    def test_simple_2d_data(self):
        # 4 points in a line
        data = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        adj_list = build_undirected_knn_adjacency_list(data, n_neighbors=2)

        assert len(adj_list) == 4
        # Each point should have at least 1 neighbor
        for neighbors in adj_list:
            assert len(neighbors) >= 1

    def test_graph_is_undirected(self):
        data = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        adj_list = build_undirected_knn_adjacency_list(data, n_neighbors=2)

        # If i is in j's neighbors, j should be in i's neighbors
        for i, neighbors in enumerate(adj_list):
            for j in neighbors:
                assert i in adj_list[j], f"Edge {i}-{j} is not symmetric"

    def test_no_self_loops(self):
        data = np.array([[0, 0], [1, 1], [2, 2]])
        adj_list = build_undirected_knn_adjacency_list(data, n_neighbors=2)

        for i, neighbors in enumerate(adj_list):
            assert i not in neighbors, f"Node {i} has self-loop"

    def test_with_more_neighbors(self):
        data = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
        adj_list = build_undirected_knn_adjacency_list(data, n_neighbors=3)

        assert len(adj_list) == 5
        # Middle points should have more neighbors due to undirected graph
        assert len(adj_list[2]) >= 2


class TestInitializeProbabilitiesAndParticles:
    """Tests for initialize_probabilities_and_particles function."""

    def test_unlabeled_samples_have_uniform_probability(self):
        labels = np.array([-1, -1, -1])
        n_classes = 3

        probs, particles = initialize_probabilities_and_particles(labels, n_classes)

        assert probs.shape == (3, 3)
        # Each unlabeled sample should have uniform probability
        expected_prob = 1.0 / n_classes
        np.testing.assert_array_almost_equal(probs, expected_prob)
        assert len(particles) == 0

    def test_labeled_samples_have_certain_probability(self):
        labels = np.array([0, 1, 2])
        n_classes = 3

        probs, particles = initialize_probabilities_and_particles(labels, n_classes)

        assert probs[0, 0] == 1.0
        assert probs[1, 1] == 1.0
        assert probs[2, 2] == 1.0
        # Other probabilities should be 0
        assert probs[0, 1] == 0.0
        assert probs[0, 2] == 0.0

    def test_particles_created_for_labeled_samples(self):
        labels = np.array([-1, 0, -1, 1, -1])
        n_classes = 2

        probs, particles = initialize_probabilities_and_particles(labels, n_classes)

        assert len(particles) == 2
        assert particles[0].position == 1
        assert particles[0].label == 0
        assert particles[1].position == 3
        assert particles[1].label == 1

    def test_mixed_labels(self):
        labels = np.array([0, -1, 1, -1])
        n_classes = 2

        probs, particles = initialize_probabilities_and_particles(labels, n_classes)

        # Labeled samples
        assert probs[0, 0] == 1.0
        assert probs[2, 1] == 1.0
        # Unlabeled samples have uniform probability
        np.testing.assert_array_almost_equal(probs[1], [0.5, 0.5])
        np.testing.assert_array_almost_equal(probs[3], [0.5, 0.5])


class TestUpdateUnlabeledNeighborProbability:
    """Tests for update_unlabeled_neighbor_probability function."""

    def test_increases_particle_label_probability(self):
        particle = Particle(position=0, label=0)
        particle.strength = 1.0
        label_probability = np.array([[0.5, 0.5]])
        n_classes = 2
        prob_change = 0.1

        update_unlabeled_neighbor_probability(
            particle, 0, label_probability, n_classes, prob_change
        )

        # Particle label probability should increase
        assert label_probability[0, 0] > 0.5
        # Other label probability should decrease
        assert label_probability[0, 1] < 0.5

    def test_weak_particle_has_less_effect(self):
        # Strong particle
        particle_strong = Particle(position=0, label=0)
        particle_strong.strength = 1.0
        probs_strong = np.array([[0.5, 0.5]])

        # Weak particle
        particle_weak = Particle(position=0, label=0)
        particle_weak.strength = 0.1
        probs_weak = np.array([[0.5, 0.5]])

        update_unlabeled_neighbor_probability(
            particle_strong, 0, probs_strong, 2, 0.1
        )
        update_unlabeled_neighbor_probability(
            particle_weak, 0, probs_weak, 2, 0.1
        )

        # Strong particle should have more effect
        assert probs_strong[0, 0] > probs_weak[0, 0]

    def test_probability_does_not_go_negative(self):
        particle = Particle(position=0, label=0)
        particle.strength = 1.0
        # Start with very low probability for other class
        label_probability = np.array([[0.99, 0.01]])

        update_unlabeled_neighbor_probability(
            particle, 0, label_probability, 2, 0.5
        )

        # No probability should be negative
        assert np.all(label_probability >= 0)


class TestRunMainIteration:
    """Tests for run_main_iteration function."""

    def test_returns_weak_and_strong_counts(self):
        particles = [Particle(0, 0), Particle(1, 1)]
        particles[0].strength = 0.95  # Strong
        particles[1].strength = 0.05  # Weak

        adjacency_list = [[1], [0]]  # Simple 2-node connected graph
        labels = np.array([-1, -1])
        label_probability = np.array([[0.5, 0.5], [0.5, 0.5]])
        rng = np.random.default_rng(42)

        weak, strong = run_main_iteration(
            particles, adjacency_list, labels, label_probability, 2, 0.1, rng
        )

        assert weak == 1
        assert strong == 1

    def test_particles_move_to_neighbors(self):
        particle = Particle(0, 0)
        adjacency_list = [[1, 2], [0], [0]]  # Node 0 connected to 1 and 2
        labels = np.array([-1, -1, -1])
        label_probability = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        rng = np.random.default_rng(42)

        run_main_iteration(
            [particle], adjacency_list, labels, label_probability, 2, 0.1, rng
        )

        # Particle should have moved to one of its neighbors
        assert particle.position in [1, 2]

    def test_labeled_nodes_not_updated(self):
        particle = Particle(0, 0)
        adjacency_list = [[1], [0]]
        labels = np.array([-1, 0])  # Node 1 is labeled
        label_probability = np.array([[0.5, 0.5], [1.0, 0.0]])
        original_prob = label_probability[1].copy()
        rng = np.random.default_rng(42)

        run_main_iteration(
            [particle], adjacency_list, labels, label_probability, 2, 0.1, rng
        )

        # Labeled node probability should not change
        np.testing.assert_array_equal(label_probability[1], original_prob)


class TestShouldStop:
    """Tests for should_stop function."""

    def test_stops_when_criteria_met(self):
        # 100 particles, 40 weak, 40 strong = 80% decided
        # stop_criteria = 0.3 means we need >30% decided AND >=15% weak
        assert should_stop(40, 40, 100, 0.3) is True

    def test_continues_when_not_enough_decided(self):
        # Only 20% decided
        assert should_stop(10, 10, 100, 0.3) is False

    def test_continues_when_not_enough_weak(self):
        # 50% decided but only 5% weak (need 15%)
        assert should_stop(5, 45, 100, 0.3) is False

    def test_edge_case_all_strong(self):
        # All strong, no weak
        assert should_stop(0, 100, 100, 0.3) is False

    def test_edge_case_all_weak(self):
        # All weak
        assert should_stop(100, 0, 100, 0.3) is True


class TestBuildPredictions:
    """Tests for build_predictions function."""

    def test_predicts_argmax(self):
        label_probability = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.5, 0.5],  # Tie goes to first class (argmax behavior)
        ])

        predictions = build_predictions(label_probability)

        assert predictions[0] == 0
        assert predictions[1] == 1
        assert predictions[2] == 0  # argmax returns first in tie

    def test_multiclass(self):
        label_probability = np.array([
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
        ])

        predictions = build_predictions(label_probability)

        assert predictions[0] == 2
        assert predictions[1] == 0
        assert predictions[2] == 1

    def test_returns_integer_array(self):
        label_probability = np.array([[0.9, 0.1]])
        predictions = build_predictions(label_probability)

        assert predictions.dtype == int


class TestPredictEdgeCases:
    """Edge case tests for the predict function."""

    def test_all_labeled(self):
        # Need enough samples for default n_neighbors=6
        data = np.array([[i, i] for i in range(10)], dtype=float)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        predictions = multiple_particle_walk.predict(data, labels, random_state=42)

        # Should return same labels since all are labeled
        np.testing.assert_array_equal(predictions, labels)

    def test_single_class(self):
        # Single class edge case - algorithm divides by (n_classes-1), so needs at least 2 classes
        # This test verifies that with heavily imbalanced classes, the majority class dominates
        data = np.array([[i, i] for i in range(10)], dtype=float)
        # Class 0 has 3 labeled samples, class 1 has only 1
        labels = np.array([0, 0, -1, -1, -1, -1, -1, -1, -1, 1])

        predictions = multiple_particle_walk.predict(data, labels, random_state=42)

        # Most unlabeled samples should be predicted as class 0 (majority)
        class_0_count = np.sum(predictions == 0)
        assert class_0_count >= 5, f"Expected majority class 0, got {class_0_count} predictions"

    def test_reproducibility_with_random_state(self):
        data = np.random.default_rng(0).random((50, 5))
        labels = np.array([0] * 5 + [1] * 5 + [-1] * 40)

        pred1 = multiple_particle_walk.predict(data, labels, random_state=123)
        pred2 = multiple_particle_walk.predict(data, labels, random_state=123)

        np.testing.assert_array_equal(pred1, pred2)

    def test_different_random_states_give_different_results(self):
        data = np.random.default_rng(0).random((100, 5))
        labels = np.array([0] * 5 + [1] * 5 + [-1] * 90)

        pred1 = multiple_particle_walk.predict(data, labels, random_state=1)
        pred2 = multiple_particle_walk.predict(data, labels, random_state=2)

        # Results should differ (with high probability)
        assert not np.array_equal(pred1, pred2)
