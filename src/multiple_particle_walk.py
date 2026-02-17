# coding=utf8
"""
Created on Sat Feb 17 16:37:38 2018

@author: Raul M. Souza

The Multiple Particle Competition and Cooperation or Multiple Particle Walk algorithm
 is a semi-supervised classification model.

It is a modified version of the original Particle Competition and Cooperation algorithm, see the reference
for more information about the original.

The difference in this version is that each labeled sample will generate multiple 
particles instead of one and there is only random walk movement.

Notes
-----
References:
    
[1] Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gonçalves; Pedrycz, Witold; Liu, Jiming, 
"Particle Competition and Cooperation in Networks for Semi-Supervised Learning," 
Knowledge and Data Engineering, IEEE Transactions on , vol.24, no.9, pp.1686,1698, Sept. 2012
doi: 10.1109/TKDE.2011.119

"""

from sklearn.neighbors import kneighbors_graph
import numpy as np 
import random


class particle:
    def __init__(self, position, label):
        self.strength = 1.0
        self.position = position
        self.label = label


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


def initialize_probabilities_and_particles(labels, n_classes):
    label_probability = np.zeros((labels.size,n_classes))
    particles = []

    for i in range(0, labels.size):
        if labels[i] == -1:
            for j in range(0,n_classes):
                label_probability[i,j] = 1.0/n_classes
        else:
            particles.append(particle(i,labels[i]))
            label_probability[i,labels[i]] = 1.0

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


def run_main_iteration(particles, adjacency_list, labels, label_probability, n_classes, prob_change):
    weak_particles = 0
    strong_particles = 0

    for i in range(0, len(particles)):
        particle_item = particles[i]

        if particle_item.strength >= 0.9:
            strong_particles+=1
        elif particle_item.strength <= 0.1:
            weak_particles+=1

        neighbor = random.choice(adjacency_list[particle_item.position])

        if labels[neighbor] == -1:
            update_unlabeled_neighbor_probability(particle_item, neighbor, label_probability, n_classes, prob_change)

        particle_item.strength = min(label_probability[neighbor,particle_item.label],1)
        particle_item.position = neighbor

    return weak_particles, strong_particles


def should_stop(weak_particles, strong_particles, total_particles, stop_criteria):
    return (weak_particles + strong_particles)>((total_particles)*stop_criteria) and weak_particles>=(total_particles*(stop_criteria/2))


def build_predictions(label_probability):
    predictions = np.zeros(len(label_probability)).astype(int)

    for i in range(0,len(label_probability)):
        predictions[i] = np.argmax(label_probability[i])

    return predictions


def predict(data, labels, n_neighbors = 6, particle_multiplier = 30, prob_change = 0.1, stop_criteria = 0.3, max_iter = 100):

    classes = np.unique(labels)
    classes = (classes[classes != -1])
    n_classes = len(classes)

    #the algorithm performs better with undirected graphs
    adjacency_list = build_undirected_knn_adjacency_list(data, n_neighbors)

    #the probabilities of each instance belonging to each target class
    label_probability, particles = initialize_probabilities_and_particles(labels, n_classes)


    #"Clone" the particles repeated times for the main loop
    particles = particles*particle_multiplier

    #Main Loop
    for _ in range(0, max_iter):
        weak_particles, strong_particles = run_main_iteration(
            particles,
            adjacency_list,
            labels,
            label_probability,
            n_classes,
            prob_change
        )
            
        #check the stop criteria    
        if should_stop(weak_particles, strong_particles, len(particles), stop_criteria):
            break

    return build_predictions(label_probability)
