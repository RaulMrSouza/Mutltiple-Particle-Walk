# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 23:03:01 2018

@author: Raul M. Souza

Multiple Particle Walk Example
"""

import multiple_particle_walk
import numpy as np
from sklearn import datasets
from sklearn import metrics

dataset = datasets.load_digits()

#take a random labeled sample from the dataset
rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(len(dataset.target)) < 0.9
labels = np.copy(dataset.target)
labels[random_unlabeled_points] = -1

predictions = multiple_particle_walk.predict(dataset.data,labels)

#print the predictions
#for i,val in enumerate(dataset.target):
    #print('predicted: '+str(predictions[i])+' expected: '+str(dataset.target[i]))

print('\nAccuracy: '+str(round(metrics.accuracy_score(dataset.target,predictions)*100,2))+'%')

#print('\nHomogeneity: '+str(round(metrics.homogeneity_score(dataset.target, predictions)*100,2))+'%')

#print('\nCompleteness: '+str(round(metrics.completeness_score(dataset.target, predictions)*100,2))+'%')

#print('\nV_measure: '+str(round(metrics.v_measure_score(dataset.target, predictions)*100,2))+'%')

#print('\nadjusted_rand_score: '+str(round(metrics.adjusted_rand_score(dataset.target, predictions)*100,2))+'%')
