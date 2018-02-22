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

from sklearn.neighbors import KDTree
import numpy as np 
import random

class particle:
    def __init__(self, position, label):
        self.strength = 1.0
        self.position = position
        self.label = label


def make_undirected_graph(tree):
    graph = []
    #append tree to new object
    for i in range(0,len(tree)):
        graph.append(list(tree[i]))
    
    #add the other adjacencies
    for i in range(0,len(tree)):
        for j in range(0,len(tree)):
            if i in tree[j]:
                graph[i].append(j)
    
    return graph


def predict(data, labels, n_neighbors = 6, particle_multiplier = 30, prob_change = 0.1, stop_criteria = 0.3, max_iter = 100):

    kdt = KDTree(data, leaf_size=30, metric='euclidean')

    tree = kdt.query(data, n_neighbors, return_distance=False) 
    
    tree = tree[:,1:n_neighbors]

    classes = np.unique(labels)
    classes = (classes[classes != -1])
    n_classes = len(classes)

    #the algorithm performs better with undirected graphs
    graph = make_undirected_graph(tree)

    #the probabilities of each instance belonging to each target class
    label_probability = np.zeros((labels.size,n_classes))

    particles = []

    # set the initial probabilities and position the particles
    for i in range(0, labels.size):
        if labels[i] == -1:
            for j in range(0,n_classes):
                label_probability[i,j] = 1.0/n_classes
        else:
            particles.append(particle(i,labels[i]))
            label_probability[i,labels[i]] = 1.0


    #"Clone" the particles repeated times for the main loop
    particles = particles*particle_multiplier

    #variables for the stop criteria
    weak_particles = 0
    strong_particles = 0

    #Main Loop
    for iter in range(0, max_iter):
        weak_particles = 0
        strong_particles = 0
        for i in range(0, len(particles)):
            if particles[i].strength >= 0.9:
                strong_particles+=1
            elif particles[i].strength <= 0.1:
                weak_particles+=1
                
            #choose a random neighbor of the particle's current position
            neighbor = random.choice(graph[particles[i].position])
            
            #if not labeled, it's label probabilities are going to be updated
            if labels[neighbor] == -1:
                #the probability change on the labels different of the particle's
                other_label_prob_change = (particles[i].strength * prob_change)/(n_classes-1)
                #The probability change on the same label of the particle's
                particle_label_prob_change = particles[i].strength * prob_change
                            
                #set the probabilities in the different labels 
                for j in range(0,n_classes):
                    if j != particles[i].label:
                        label_probability[neighbor,j] -= other_label_prob_change
                        #if with the update the probability is lesser than zero
                        #set it to zero and add the difference in the particle's label probability change
                        if label_probability[neighbor,j] < 0:
                            particle_label_prob_change += label_probability[neighbor,j]
                            label_probability[neighbor,j] = 0
                        #update the probability of the particle's label
                        label_probability[neighbor,particles[i].label] += particle_label_prob_change
            #the strength is set to the label probability of the same class
            particles[i].strength = min(label_probability[neighbor,particles[i].label],1)
            #update the particle's new position
            particles[i].position = neighbor
            
        #check the stop criteria    
        if (weak_particles + strong_particles)>((len(particles))*stop_criteria) and weak_particles>=(len(particles)*(stop_criteria/2)):
            break

    predictions = np.zeros(len(label_probability)).astype(int)

    for i in range(0,len(label_probability)):
        predictions[i] = np.argmax(label_probability[i])

    return predictions






